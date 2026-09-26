"""Tests for the spectral PE convection bridge — CMT plumbing.

Covers:
- vordiv_from_uv_3d: round-trip with uv_from_vordiv_3d on a known
  solid-body rotation; correctness on an analytical ζ=2Ω sin φ field.
- _make_spectral_pe_convection: bridge produces zero vor/div tendencies
  for non-CMT schemes (SBM) and propagates CMT correctly for CMT-capable
  schemes (Zhang-McFarlane).
- Differentiability: jax.grad through the bridge succeeds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    sh_synthesis_3d,
    uv_from_vordiv_3d,
    vordiv_from_uv_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    ZhangMcFarlaneConfig,
    KainFritschConfig,
    EmanuelConfig,
    TiedtkeConfig,
    BechtoldConfig,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm.core.field import Field


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    """T21 Gaussian grid for fast tests."""
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def sigma_coord():
    """10-level sigma coordinate."""
    return create_sigma_coordinate(10)


@pytest.fixture(scope="module")
def rest_state(grid, sigma_coord):
    """Isothermal 300 K rest state, no perturbation."""
    return isothermal_rest_state_spectral(
        grid, sigma_coord, perturbation_amplitude=0.0,
    )


def _state_with_tracers(state, tracers):
    """Attach a ``tracers`` dict to a ``SpectralHydrostaticState``.

    ``SpectralHydrostaticState`` now natively carries an optional
    ``tracers`` field (default ``None``) so this just wraps the
    standard ``_replace`` to keep call sites short.
    """
    return state._replace(tracers=tracers)


def _state_with_sheared_winds(rest_state, grid, sigma_coord, u_top=20.0, u_sfc=2.0):
    """Inject a vertically sheared solid-body wind into the rest state.

    Builds a per-level zonal vorticity ``2 * u_amp(σ) * sin(lat) / a``
    where ``u_amp`` ramps linearly from ``u_sfc`` at the surface to
    ``u_top`` at the model top.  Vertical shear is required for the
    Gregory CMT closure to produce a non-zero tendency (the closure
    annihilates uniform-wind columns by design).
    """
    a = grid.radius
    nlev = sigma_coord.n_levels
    # u_amp(k) — linear ramp from u_sfc at surface (k=nlev-1) to u_top at top (k=0).
    sigma_full = sigma_coord.sigma_full
    u_amp_k = u_top + (u_sfc - u_top) * (1.0 - sigma_full)   # (nlev,)

    # Solid-body relative vorticity per level = 2 u_amp(k) sin(lat) / a.
    sin_lat_2d = jnp.sin(grid.lat[:, None]) * jnp.ones(
        (grid.n_lat, grid.n_lon), dtype=jnp.float64,
    )
    vor_grid_2d = 2.0 * sin_lat_2d / a   # (n_lat, n_lon)
    vor_hat_2d = sh_analysis(grid, vor_grid_2d)              # (n_sh,)
    vor_hat_3d = vor_hat_2d[:, None] * u_amp_k[None, :]      # (n_sh, nlev)
    vor_hat_3d = vor_hat_3d.astype(rest_state.vor_hat.data.dtype)
    return rest_state._replace(
        vor_hat=rest_state.vor_hat.replace(data=vor_hat_3d),
    )


# ---------------------------------------------------------------------------
# vordiv_from_uv_3d — basic mathematical correctness
# ---------------------------------------------------------------------------

class TestVordivFromUV3D:
    def test_shapes(self, grid):
        nlev = 4
        u = jnp.zeros((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        v = jnp.zeros_like(u)
        vor_hat, div_hat = vordiv_from_uv_3d(grid, u, v)
        assert vor_hat.shape == (grid.n_sh, nlev)
        assert div_hat.shape == (grid.n_sh, nlev)
        assert vor_hat.dtype == jnp.complex128
        assert div_hat.dtype == jnp.complex128

    def test_zero_field_zero_spectral(self, grid):
        nlev = 3
        u = jnp.zeros((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        v = jnp.zeros_like(u)
        vor_hat, div_hat = vordiv_from_uv_3d(grid, u, v)
        assert float(jnp.max(jnp.abs(vor_hat))) < 1e-12
        assert float(jnp.max(jnp.abs(div_hat))) < 1e-12

    def test_solid_body_rotation_recovers_vor(self, grid):
        """For u = u0 cos φ, v = 0 → ζ = 2 u0 sin φ / a, D = 0."""
        nlev = 2
        a = grid.radius
        u0 = 20.0
        cos_lat = grid.cos_lat[:, None, None]   # (n_lat, 1, 1)

        u = jnp.broadcast_to(
            (u0 * grid.cos_lat[:, None])[:, :, None],
            (grid.n_lat, grid.n_lon, nlev),
        ).astype(jnp.float64)
        v = jnp.zeros_like(u)

        vor_hat, div_hat = vordiv_from_uv_3d(grid, u, v)

        # Expected analytical ζ = 2 u0 sin φ / a (per level).
        expected_vor_grid = (
            2.0 * u0 * jnp.sin(grid.lat[:, None]) / a
        ) * jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
        expected_vor_hat = sh_analysis(grid, expected_vor_grid)

        # Compare the n>=1 modes only (n=0 mode of vor is identically zero
        # by both routes, but float roundoff can leak into it).
        for k in range(nlev):
            assert bool(jnp.allclose(
                vor_hat[1:, k], expected_vor_hat[1:],
                atol=1e-8, rtol=1e-6,
            )), f"Solid-body vor mismatch at level {k}"
            # Divergence should be (numerically) zero everywhere.
            assert float(jnp.max(jnp.abs(div_hat[:, k]))) < 1e-8

    def test_zonal_wave_divergence(self, grid):
        """For u = u0 cos(λ), v = 0 → analytical div = -u0 sin(λ) / (a cos φ).

        Sanity-check the spectral div/curl signature on a non-trivial
        wave-1 zonal flow.  We synthesize the result back to the grid
        and compare to the analytical expression.
        """
        nlev = 1
        a = grid.radius
        u0 = 5.0
        lon = grid.lon[None, :, None]   # (1, n_lon, 1)
        cos_phi_3d = grid.cos_lat[:, None, None]
        cos_phi_safe = jnp.clip(cos_phi_3d, 0.05, None)

        u = u0 * jnp.cos(lon) * jnp.ones(
            (grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64,
        )
        v = jnp.zeros_like(u)

        _, div_hat = vordiv_from_uv_3d(grid, u, v)
        div_grid = sh_synthesis_3d(grid, div_hat)

        # Analytical: ∇·(u, 0) = (1/(a cos φ)) ∂u/∂λ = -u0 sin(λ)/(a cos φ).
        expected_div = -u0 * jnp.sin(lon) / (a * cos_phi_safe) * jnp.ones_like(u)

        # Mask out polar caps (analytical solution diverges as 1/cos φ).
        mask = grid.cos_lat > 0.4
        assert bool(jnp.allclose(
            div_grid[mask, :], expected_div[mask, :],
            atol=1e-6, rtol=5e-3,
        )), "Wave-1 zonal divergence does not match analytical form"

    def test_jit_compatible(self, grid):
        # The GaussianGrid mixes Python ints and JAX arrays, so it
        # cannot be a JIT argument directly.  Mirror the spectral PE
        # pattern of closure-capturing the grid and JIT-compiling the
        # wrapper.
        nlev = 2
        u = jnp.ones((grid.n_lat, grid.n_lon, nlev), dtype=jnp.float64)
        v = jnp.zeros_like(u)

        @jax.jit
        def f(u, v):
            return vordiv_from_uv_3d(grid, u, v)

        vor_hat, div_hat = f(u, v)
        assert vor_hat.shape == (grid.n_sh, nlev)
        assert div_hat.shape == (grid.n_sh, nlev)

    def test_grad_flows(self, grid):
        nlev = 2

        def loss(u_amp):
            u = u_amp * jnp.broadcast_to(
                grid.cos_lat[:, None, None],
                (grid.n_lat, grid.n_lon, nlev),
            )
            v = jnp.zeros_like(u)
            vor_hat, _ = vordiv_from_uv_3d(grid, u, v)
            return jnp.sum(jnp.abs(vor_hat) ** 2).real

        g = jax.grad(loss)(jnp.array(2.0))
        assert bool(jnp.isfinite(g))
        # Sensitivity to u_amp must be positive (ζ ~ u_amp).
        assert float(g) > 0.0


# ---------------------------------------------------------------------------
# Spectral PE bridge — CMT propagation
# ---------------------------------------------------------------------------

class TestSpectralPECMT:
    def test_non_cmt_scheme_yields_zero_vor_div(self, grid, sigma_coord, rest_state):
        """SBM is not CMT-capable — vor/div tendencies must remain zero."""
        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="sbm"),
            model_type="spectral_pe", dt=300.0,
        )
        tendencies, _ = physics_fn(rest_state, grid, sigma_coord)
        assert float(jnp.max(jnp.abs(tendencies.vor_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(tendencies.div_hat.data))) == 0.0

    def test_zm_runs_with_vapour_and_cloud_tracers_and_refuses_without_cloud(
        self, grid, sigma_coord, rest_state,
    ):
        """CAM6 Zhang-McFarlane emits a SIGNED net rain-flux divergence whose
        column integral is the surface rain.  The bridge books only dq_v and
        dq_c and lets the rain leave the column (the hydrostatic bridge's
        surface route), so it needs BOTH tracers; without q_c the detrained
        condensate would vanish unrecorded, and that is refused loudly."""
        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="zhang_mcfarlane"),
            model_type="spectral_pe", dt=300.0,
        )
        zeros = jnp.zeros((grid.n_lat, grid.n_lon, sigma_coord.n_levels))
        q_v = Field(data=zeros, name="q_v", dims=("lat", "lon", "level"), units="kg/kg")
        q_c = Field(data=zeros, name="q_c", dims=("lat", "lon", "level"), units="kg/kg")
        tendencies, _ = physics_fn(
            _state_with_tracers(rest_state, {"q_v": q_v, "q_c": q_c}), grid, sigma_coord)
        assert set(tendencies.tracers) == {"q_v", "q_c"}
        for k in ("q_v", "q_c"):
            assert bool(jnp.all(jnp.isfinite(tendencies.tracers[k].data)))
        with pytest.raises(ValueError, match="must carry both tracers"):
            physics_fn(_state_with_tracers(rest_state, {"q_v": q_v}), grid, sigma_coord)

    def test_tiedtke_with_cape_and_winds_yields_nonzero_cmt(
        self, grid, sigma_coord, rest_state,
    ):
        """Inject CAPE + winds → the CMT scheme should fire and produce non-zero CMT.

        Constructs a state with non-zero u (solid-body rotation) and
        attaches a tracers dict with q_v close to saturation.  The
        bridge then sees: q_v > 0, T positive CAPE structure, u > 0 →
        plume mass flux > 0 → non-zero du_dt_conv → non-zero
        d(vor_hat) and d(div_hat).
        """
        # Build a state with non-zero, vertically-sheared u (necessary
        # because Gregory CMT vanishes for uniform-wind columns).
        state_w_u = _state_with_sheared_winds(
            rest_state, grid, sigma_coord, u_top=20.0, u_sfc=2.0,
        )

        # Build a CAPE-positive temperature profile in grid space.
        nlev = sigma_coord.n_levels
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)
        state_w_uT = state_w_u._replace(
            T_hat=state_w_u.T_hat.replace(data=T_hat),
        )

        # Build a near-saturated q_v field; attach as a tracers dict to
        # the spectral state via the duck-typed wrapper.  Match the
        # bridge's expected raw-array shape (state.tracers["q_v"].data,
        # shape (n_lat, n_lon, nlev) — the bridge calls .reshape so the
        # leading 2D layout is fine).
        from legoesm.thermo import saturation_mixing_ratio
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.5)
        q_v_grid = rh_profile[None, None, :] * q_sat
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )
        state_with_tracers = _state_with_tracers(state_w_uT, {"q_v": q_v_field})

        # Tiedtke: the CMT-capable scheme this bridge can run (ZM's net rain
        # flux needs a surface sink this bridge lacks; see the refusal test).
        physics_fn = make_convection_physics(
            ConvectionConfig(
                scheme="tiedtke",
                tiedtke=TiedtkeConfig(precip_efficiency=0.0),
            ),
            model_type="spectral_pe", dt=300.0,
        )

        tendencies, _ = physics_fn(state_with_tracers, grid, sigma_coord)

        # Plume should fire somewhere — vor/div tendencies must be
        # non-zero in at least one mode/level.
        assert float(jnp.max(jnp.abs(tendencies.T_hat.data))) > 0.0, (
            "Tiedtke should produce non-zero T tendencies on a CAPE column"
        )
        assert float(jnp.max(jnp.abs(tendencies.vor_hat.data))) > 0.0, (
            "Tiedtke should produce non-zero vor_hat tendencies via CMT"
        )
        assert float(jnp.max(jnp.abs(tendencies.div_hat.data))) > 0.0, (
            "Tiedtke should produce non-zero div_hat tendencies via CMT"
        )

    def test_tiedtke_responds_to_moisture_convergence(
        self, grid, sigma_coord, rest_state,
    ):
        """Tiedtke's deep closure consumes ``moisture_convergence``.  Two
        spectral states with the same temperature/q_v but different
        wind divergences produce different ``q_v`` flux divergences →
        different Tiedtke tendencies.

        The pre-fix bridge passed ``mc_col=zeros`` regardless of state,
        so this test would have shown identical tendencies.  With the
        spectral-MC plumbing the two columns now differ.
        """
        nlev = sigma_coord.n_levels

        # CAPE-positive temperature profile.
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)

        from legoesm.thermo import saturation_mixing_ratio
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.5)
        q_v_grid = rh_profile[None, None, :] * q_sat
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )

        state_no_div = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
        )
        state_no_div = _state_with_tracers(state_no_div, {"q_v": q_v_field})

        # Wave-1 divergence as in test_kf_w_grid_responds_to_divergence.
        a = grid.radius
        u0 = 5.0
        lon = grid.lon[None, :]
        cos_lat_safe = jnp.clip(grid.cos_lat[:, None], 0.05, None)
        div_grid_2d = -u0 * jnp.sin(lon) / (a * cos_lat_safe)
        div_hat_2d = sh_analysis(grid, div_grid_2d.astype(jnp.float64))
        div_hat_3d = jnp.broadcast_to(
            div_hat_2d[:, None], (grid.n_sh, nlev),
        ).astype(rest_state.div_hat.data.dtype)
        state_w_div = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            div_hat=rest_state.div_hat.replace(data=div_hat_3d),
        )
        state_w_div = _state_with_tracers(state_w_div, {"q_v": q_v_field})

        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="tiedtke"),
            model_type="spectral_pe", dt=300.0,
        )

        tend_a, prog_a = physics_fn(state_no_div, grid, sigma_coord)
        tend_b, prog_b = physics_fn(state_w_div, grid, sigma_coord)

        # The cloud-base mass-flux carry should differ — Tiedtke's
        # closure ``M_b ∝ MC`` reads the spectral-derived MC, which is
        # zero in state_a (no div) and non-zero in state_b (wave-1).
        a_carry_max = float(jnp.max(jnp.abs(prog_a)))
        b_carry_max = float(jnp.max(jnp.abs(prog_b)))
        # b_carry should differ from a_carry — proves the MC plumbing
        # is active.
        max_diff = float(jnp.max(jnp.abs(prog_a - prog_b)))
        assert max_diff > 0.0, (
            "Tiedtke carry should respond to spectral-derived MC "
            f"(no-div max={a_carry_max}, w-div max={b_carry_max})"
        )

    def test_kf_w_grid_responds_to_divergence(
        self, grid, sigma_coord, rest_state,
    ):
        """KF's bridge derives ``w_grid`` from spectral divergence.  Two
        spectral states identical in everything except ``div_hat``
        should produce different KF tendencies when CAPE > 0 — the
        zero-div state and the wave-1 zonal-divergence state must
        differ at the diagnostic-mass-flux carry.
        """
        nlev = sigma_coord.n_levels

        # CAPE-positive temperature profile (idealised tropical sounding).
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)

        from legoesm.thermo import saturation_mixing_ratio
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        rh_profile = jnp.where(sigma_full > 0.7, 0.95, 0.5)
        q_v_grid = rh_profile[None, None, :] * q_sat
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )

        state_no_div = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
        )
        state_no_div = _state_with_tracers(state_no_div, {"q_v": q_v_field})

        # Wave-1 divergence: D ∝ -sin(λ) / cos(φ).
        a = grid.radius
        u0 = 5.0
        lon = grid.lon[None, :]
        cos_lat_safe = jnp.clip(grid.cos_lat[:, None], 0.05, None)
        div_grid_2d = -u0 * jnp.sin(lon) / (a * cos_lat_safe)
        div_hat_2d = sh_analysis(grid, div_grid_2d.astype(jnp.float64))
        div_hat_3d = jnp.broadcast_to(
            div_hat_2d[:, None], (grid.n_sh, nlev),
        ).astype(rest_state.div_hat.data.dtype)
        state_w_div = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            div_hat=rest_state.div_hat.replace(data=div_hat_3d),
        )
        state_w_div = _state_with_tracers(state_w_div, {"q_v": q_v_field})

        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="kain_fritsch"),
            model_type="spectral_pe", dt=300.0,
        )

        tend_a, prog_a = physics_fn(state_no_div, grid, sigma_coord)
        tend_b, prog_b = physics_fn(state_w_div, grid, sigma_coord)

        # The cloud-base mass flux carry (packed at [:, -1]) should
        # differ between the two states because divergence shifts the
        # KF trigger via w_grid_at_lcl.
        a_carry = prog_a[:, -1]
        b_carry = prog_b[:, -1]
        max_diff = float(jnp.max(jnp.abs(a_carry - b_carry)))
        assert max_diff > 0.0, (
            "KF carry should respond to divergence-derived w_grid"
        )

    def test_full_dycore_step_with_tracers_and_convection(
        self, grid, sigma_coord, rest_state,
    ):
        """End-to-end integration: spectral PE dycore stepping with a
        tracer-aware state AND a convection physics_fn, all wired
        through ``step_with_physics``.  This exercises:

        * ``state.tracers`` survives the RK stepping (pytree match).
        * The convection bridge reads ``state.tracers["q_v"]`` natively.
        * The orchestrator/dycore RHS combine cleanly with the
          tracer-aware tendency from ``spectral_pe_tendencies``.

        Pre-fix this would have raised "Expected dict, got None" inside
        the SSP-RK3 ``jax.tree.map`` because the tendency state's
        ``tracers`` field was None while the input state's was a dict.
        """
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig, SpectralPrimitiveEquationModel,
        )
        nlev = sigma_coord.n_levels

        # CAPE-positive temperature profile + near-saturated q_v.
        sigma_full = sigma_coord.sigma_full
        T_col = 300.0 * jnp.power(jnp.clip(sigma_full, 0.05, None), 0.19)
        T_col = jnp.maximum(T_col, 200.0)
        T_grid = jnp.broadcast_to(
            T_col[None, None, :], (grid.n_lat, grid.n_lon, nlev),
        )
        T_hat = sh_analysis_3d(grid, T_grid)

        from legoesm.thermo import saturation_mixing_ratio
        p_full_3d = jnp.broadcast_to(
            (sigma_full * 1e5)[None, None, :],
            (grid.n_lat, grid.n_lon, nlev),
        )
        q_sat = saturation_mixing_ratio(T_grid, p_full_3d)
        q_v_grid = 0.7 * q_sat
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )
        state_with_T_and_q = rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            tracers={"q_v": q_v_field},
        )

        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="tiedtke"),
            model_type="spectral_pe", dt=300.0,
        )

        config = SpectralPEConfig(
            hyperdiff_coeff=1.0 / (4.0 * 3600.0 * (
                grid.n_max * (grid.n_max + 1) / grid.radius ** 2
            ) ** 2),
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma_coord, config)
        new_state = model.step(
            state_with_T_and_q, dt=300.0, physics_fn=physics_fn,
        )

        # With the new tracer-aware dycore, Tiedtke's q_v sink is
        # actually applied across the SSP-RK stages.  We expect the
        # boundary-layer q_v to *drop* slightly (convection consumes
        # vapor) — but stay finite and bounded by the initial q_v.
        assert new_state.tracers is not None
        assert "q_v" in new_state.tracers
        new_qv = new_state.tracers["q_v"].data
        assert bool(jnp.all(jnp.isfinite(new_qv)))
        # Tracer remains positive (q_v >= 0) — physical realism check.
        assert float(jnp.min(new_qv)) > -1e-12
        # The change is bounded by an aggressive upper limit on
        # convective drying over 1 step at dt=300s with q_sat ~ 1.4e-2:
        # ~0.5% of the initial field magnitude.
        max_change = float(jnp.max(jnp.abs(new_qv - q_v_grid)))
        assert max_change < 5e-3, (
            f"q_v evolution exceeded tolerance: max change {max_change}"
        )
        # All spectral fields are finite (no NaN from tracer plumbing).
        assert bool(jnp.all(jnp.isfinite(new_state.vor_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.div_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.T_hat.data)))
        assert bool(jnp.all(jnp.isfinite(new_state.lnps_hat.data)))

    def test_grad_through_bridge(self, grid, sigma_coord, rest_state):
        """jax.grad through the spectral PE convection bridge succeeds.

        Uses a real-valued amplitude parameter as the differentiation
        target so the loss is real → real (sidestepping the complex /
        holomorphic-input dance with a complex T_hat).
        """
        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="tiedtke",
                             tiedtke=TiedtkeConfig(precip_efficiency=0.0)),
            model_type="spectral_pe", dt=300.0,
        )

        T_hat_base = rest_state.T_hat.data

        def loss(scale):
            new_T = scale * T_hat_base
            new_state = rest_state._replace(
                T_hat=rest_state.T_hat.replace(data=new_T),
            )
            tend, _ = physics_fn(new_state, grid, sigma_coord)
            return (
                jnp.sum(jnp.abs(tend.T_hat.data) ** 2).real
                + jnp.sum(jnp.abs(tend.vor_hat.data) ** 2).real
                + jnp.sum(jnp.abs(tend.div_hat.data) ** 2).real
            )

        g = jax.grad(loss)(jnp.array(1.0))
        assert bool(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# All 5 profile-prognostic schemes must run on spectral PE.  Pre-fix the
# spectral-PE branch of ``_make_spectral_pe_convection`` had a flat elif
# chain that fell through to a catch-all ``else`` passing ``u`` and ``v``
# to any non-stochastic / non-w-grid / non-MC scheme — Emanuel's leaf
# does not accept ``u``/``v``, so the kernel raised ``TypeError`` at
# call time.  This regression pins all five schemes through the bridge.
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="module",
    params=[
        # zhang_mcfarlane is absent by design: this fixture's state is q_v-only
        # and ZM needs a q_c tracer for its detrained condensate (its signed
        # net rain flux takes the surface route); it is pinned through the
        # bridge in tests/unit/test_spectral_zm_rain_surface_route.py.
        ("kain_fritsch", "kain_fritsch", KainFritschConfig),
        ("emanuel", "emanuel", EmanuelConfig),
        ("tiedtke", "tiedtke", TiedtkeConfig),
        ("bechtold", "bechtold", BechtoldConfig),
    ],
    ids=["kain_fritsch", "emanuel", "tiedtke", "bechtold"],
)
def profile_scheme_config(request):
    """Yield a (name, ConvectionConfig) pair for each new scheme."""
    name, scheme_field, scheme_cls = request.param
    # #929: this dispatch / finite-output regression shares one condensate-less
    # (``q_v``-only) spectral state across all five schemes.  Default Bechtold
    # now emits an in-updraft rain split (``precip_efficiency=0.7``) that the
    # spectral bridge REQUIRES a ``q_c``/``q_r`` tracer to receive — otherwise
    # it raises loudly rather than silently leaking the un-booked rain water.
    # Pin ``precip_efficiency=0.0`` here so Bechtold exercises its pre-#929
    # no-split dispatch path on this condensate-less state; the rain-split
    # routing and the raise are covered by
    # ``test_convection_rain_split_bridges.py``.
    if scheme_cls is BechtoldConfig:
        kwargs = {scheme_field: scheme_cls(precip_efficiency=0.0)}
    else:
        kwargs = {scheme_field: scheme_cls()}
    return name, ConvectionConfig(scheme=name, **kwargs)


class TestSpectralPEAllProfileSchemes:
    """Each new profile-prognostic scheme must run end-to-end on a
    Gaussian spectral-PE state without raising and produce finite
    tendencies.  This catches the spectral-PE Emanuel dispatch bug
    (TypeError: emanuel_convection() got an unexpected keyword
    argument 'u') and prevents a similar regression for the other
    schemes."""

    def _state_with_cape_and_qv(self, grid, sigma_coord, rest_state):
        """Build a CAPE-positive state with near-saturated q_v so the
        new schemes have a non-trivial trigger surface to act on."""
        from legoesm.thermo import saturation_mixing_ratio

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
        q_v_grid = rh_profile[None, None, :] * q_sat
        q_v_field = Field(
            data=q_v_grid, name="q_v",
            dims=("lat", "lon", "level"), units="kg/kg",
        )
        return rest_state._replace(
            T_hat=rest_state.T_hat.replace(data=T_hat),
            tracers={"q_v": q_v_field},
        )

    def test_bridge_call_succeeds(
        self, profile_scheme_config, grid, sigma_coord, rest_state,
    ):
        """Bridge runs end-to-end without raising for each scheme."""
        scheme_name, cfg = profile_scheme_config
        state = self._state_with_cape_and_qv(grid, sigma_coord, rest_state)
        physics_fn = make_convection_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        # The bug surfaced as TypeError at call time — exercise the
        # full bridge path including the elif dispatch.
        tend, prog = physics_fn(state, grid, sigma_coord)
        assert tend is not None
        assert tend.T_hat.data.shape == state.T_hat.data.shape

    def test_finite_output(
        self, profile_scheme_config, grid, sigma_coord, rest_state,
    ):
        """Tendencies are finite for each scheme on spectral PE."""
        scheme_name, cfg = profile_scheme_config
        state = self._state_with_cape_and_qv(grid, sigma_coord, rest_state)
        physics_fn = make_convection_physics(
            cfg, model_type="spectral_pe", dt=300.0,
        )
        tend, prog = physics_fn(state, grid, sigma_coord)
        assert bool(jnp.all(jnp.isfinite(tend.T_hat.data))), (
            f"{scheme_name}: T_hat tendency has NaN/Inf"
        )
        assert bool(jnp.all(jnp.isfinite(tend.vor_hat.data)))
        assert bool(jnp.all(jnp.isfinite(tend.div_hat.data)))
        assert bool(jnp.all(jnp.isfinite(tend.lnps_hat.data)))
        if prog is not None:
            # Profile-prognostic carry can be a dict (Bechtold) or a
            # raw array (the rest).  In both cases the array(s) inside
            # must be finite — the dict carries
            # ``conv_prog_profile``/``conv_stoch_state``/``prng_key``.
            if isinstance(prog, dict):
                for k, v in prog.items():
                    if v is not None:
                        assert bool(jnp.all(jnp.isfinite(v))), (
                            f"{scheme_name}: prog[{k}] has NaN/Inf"
                        )
            else:
                assert bool(jnp.all(jnp.isfinite(prog)))

    def test_emanuel_spectral_pe_does_not_receive_winds(
        self, grid, sigma_coord, rest_state,
    ):
        """Direct regression for the original Codex finding.

        Pre-fix:
            TypeError: emanuel_convection() got an unexpected
            keyword argument 'u'
        because the spectral-PE bridge's catch-all ``else`` branch
        passed ``u``/``v`` to non-CMT, non-MC, non-w-grid profile
        schemes — Emanuel's leaf has no wind kwargs.
        """
        state = self._state_with_cape_and_qv(grid, sigma_coord, rest_state)
        physics_fn = make_convection_physics(
            ConvectionConfig(scheme="emanuel", emanuel=EmanuelConfig()),
            model_type="spectral_pe", dt=300.0,
        )
        # The original bug raised inside the JIT-traced kernel call.
        tend, _ = physics_fn(state, grid, sigma_coord)
        # And the resulting T tendency must be finite (no NaN).
        assert bool(jnp.all(jnp.isfinite(tend.T_hat.data)))
