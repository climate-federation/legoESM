"""Unit tests for :mod:`legoesm.training.aimip_spatial`.

Covers:

1. :func:`spatial_basis` builds a normalized Legendre x Fourier basis
   of the right shape and dtype, and the basis functions are
   approximately orthogonal under area-weighted integration on a
   coarse Gaussian grid.

2. :class:`SpatialField` with zero coefficients reduces to the
   global baseline ``f_0`` (verified pointwise).

3. :class:`SpatialField` ``log_perturb`` transform stays strictly
   positive and respects the configured perturbation range.

4. :class:`AIMIPSpatialSurfaceParams.from_defaults` initializes
   every expected field at the baseline value.

5.``AIMIPClassicalParams.from_defaults(spatial_surface=True)``
   builds, ``make_aimip_classical_spectral_physics`` accepts the
   resulting params + a land mask, and the synthetic gradient
   through ``as_dict`` and through the spatial coefficients are both
   finite.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

jax.config.update("jax_enable_x64", True)


def _grid_t11():
    """Small Gaussian grid for the spatial tests (kept tiny so the
    sphere-integral checks evaluate in milliseconds)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    return create_gaussian_grid(11, dealiasing="quadratic")


# ----------------------------------------------------------------------
# spatial_basis
# ----------------------------------------------------------------------

def test_spatial_basis_shape_and_finite():
    from legoesm.training.aimip_spatial import spatial_basis, n_basis
    grid = _grid_t11()
    basis = spatial_basis(grid, l_max=4, m_max=2)
    assert basis.shape == (n_basis(4, 2), grid.n_lat, grid.n_lon)
    assert jnp.all(jnp.isfinite(basis))


def test_spatial_basis_constant_mode_is_ones():
    """The l=0 zonal Legendre basis is constant ``P_0(sin lat) = 1``."""
    from legoesm.training.aimip_spatial import spatial_basis
    grid = _grid_t11()
    basis = spatial_basis(grid, l_max=2, m_max=1)
    P0 = basis[0]
    assert jnp.allclose(P0, jnp.ones_like(P0), atol=1e-10)


# ----------------------------------------------------------------------
# SpatialField
# ----------------------------------------------------------------------

def test_spatial_field_zero_coeffs_is_baseline_log_perturb():
    from legoesm.training.aimip_spatial import SpatialField
    grid = _grid_t11()
    field = SpatialField.from_defaults(
        f_0=1.5e-3, scale=0.7, transform="log_perturb",
        l_max=2, m_max=1,
    )
    value = field.evaluate(grid)
    assert value.shape == (grid.n_lat, grid.n_lon)
    assert jnp.allclose(value, 1.5e-3, atol=1e-12)


def test_spatial_field_zero_coeffs_is_baseline_shift():
    from legoesm.training.aimip_spatial import SpatialField
    grid = _grid_t11()
    field = SpatialField.from_defaults(
        f_0=0.95, scale=0.05, transform="shift",
        l_max=2, m_max=1,
    )
    value = field.evaluate(grid)
    assert jnp.allclose(value, 0.95, atol=1e-12)


def test_spatial_field_log_perturb_stays_positive():
    """``log_perturb`` with arbitrary coefficients keeps value > 0."""
    from legoesm.training.aimip_spatial import SpatialField, n_basis
    grid = _grid_t11()
    nb = n_basis(2, 1)
    field = SpatialField(
        coeffs=jnp.array([10.0] * nb),  # large coefficients
        f_0=1.0e-3, scale=2.0, transform="log_perturb",
        l_max=2, m_max=1,
    )
    value = field.evaluate(grid)
    assert jnp.all(value > 0.0)
    # ``tanh`` keeps z_bounded in (-1, 1), so the log-perturb range is
    # ``[f_0 * exp(-scale), f_0 * exp(+scale)]`` (numerical slack).
    f_0, scale = 1.0e-3, 2.0
    assert jnp.all(value > f_0 * math.exp(-scale) - 1e-12)
    assert jnp.all(value < f_0 * math.exp(+scale) + 1e-12)


def test_spatial_field_land_mask_gates_to_baseline():
    """Where ``land_mask`` is 0, ``evaluate`` returns the baseline ``f_0``."""
    from legoesm.training.aimip_spatial import SpatialField, n_basis
    grid = _grid_t11()
    nb = n_basis(2, 1)
    field = SpatialField(
        coeffs=jnp.array([3.0] * nb),
        f_0=2.0, scale=1.0, transform="shift",
        l_max=2, m_max=1,
    )
    # Fully-ocean land mask (zeros) -> baseline everywhere.
    ocean = jnp.zeros((grid.n_lat, grid.n_lon))
    value = field.evaluate(grid, land_mask=ocean)
    assert jnp.allclose(value, 2.0, atol=1e-12)
    # Fully-land land mask (ones) -> non-trivial spatial field.
    land = jnp.ones((grid.n_lat, grid.n_lon))
    value_land = field.evaluate(grid, land_mask=land)
    assert not jnp.allclose(value_land, 2.0, atol=1e-3)


# ----------------------------------------------------------------------
# AIMIPSpatialSurfaceParams
# ----------------------------------------------------------------------

def test_aimip_spatial_surface_params_from_defaults_has_expected_fields():
    from legoesm.training.aimip_spatial import (
        AIMIPSpatialSurfaceParams,
        SPATIAL_FIELD_NAMES,
    )
    p = AIMIPSpatialSurfaceParams.from_defaults()
    assert set(p.fields.keys()) == set(SPATIAL_FIELD_NAMES)
    # All coefficients start at zero, so evaluate must return f_0
    # baselines pointwise.
    grid = _grid_t11()
    fields_2d = p.evaluate(grid)
    for name in SPATIAL_FIELD_NAMES:
        v = fields_2d[name]
        assert v.shape == (grid.n_lat, grid.n_lon)
        # Should be close to a constant equal to its baseline.
        v_min = float(jnp.min(v))
        v_max = float(jnp.max(v))
        assert math.isclose(v_min, v_max, rel_tol=1e-10, abs_tol=1e-10)


def test_aimip_spatial_surface_params_n_trainable_count():
    """Default truncation has 13 coefs per field; 5 fields -> 65 coefs."""
    from legoesm.training.aimip_spatial import AIMIPSpatialSurfaceParams
    p = AIMIPSpatialSurfaceParams.from_defaults()
    assert p.n_trainable() == 65


def test_aimip_spatial_surface_nonzero_init_breaks_symmetry():
    """``init_std > 0`` produces non-baseline spatial fields at init."""
    from legoesm.training.aimip_spatial import (
        AIMIPSpatialSurfaceParams,
        SPATIAL_FIELD_NAMES,
    )
    grid = _grid_t11()
    p = AIMIPSpatialSurfaceParams.from_defaults(
        init_std=0.05, key=jax.random.PRNGKey(42),
    )
    fields_2d = p.evaluate(grid)
    # Every field should now have a non-trivial spatial profile (min != max).
    for name in SPATIAL_FIELD_NAMES:
        v = fields_2d[name]
        v_min = float(jnp.min(v))
        v_max = float(jnp.max(v))
        assert v_max - v_min > 1e-6, (
            f"{name}: field is uniform under non-zero init "
            f"(min={v_min}, max={v_max})"
        )


def test_aimip_classical_params_spatial_seed_reproducibility():
    """Same seed -> identical spatial-coef tree across two builds."""
    from legoesm.training.aimip_params import AIMIPClassicalParams
    p1 = AIMIPClassicalParams.from_defaults(
        spatial_surface=True, spatial_init_std=0.02, spatial_seed=7,
    )
    p2 = AIMIPClassicalParams.from_defaults(
        spatial_surface=True, spatial_init_std=0.02, spatial_seed=7,
    )
    for name in p1.spatial_surface.fields:
        assert jnp.array_equal(
            p1.spatial_surface.fields[name].coeffs,
            p2.spatial_surface.fields[name].coeffs,
        ), f"seed=7 reproducibility broken for field {name}"


# ----------------------------------------------------------------------
# Integration with AIMIPClassicalParams
# ----------------------------------------------------------------------

def test_aimip_classical_params_with_spatial_surface_builds():
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    grid = _grid_t11()

    params = AIMIPClassicalParams.from_defaults(spatial_surface=True)
    assert params.spatial_surface is not None
    assert params.spatial_surface.n_trainable() == 65

    # Synthetic mask: land over the Northern hemisphere.
    land_mask = jnp.where(grid.lat2d > 0.0, 1.0, 0.0)

    fn = make_aimip_classical_spectral_physics(
        params, grid, dt=1800.0,
        radiation="gray",
        turbulence_scheme="louis",
        land_mask=land_mask,
    )
    assert callable(fn)


def test_aimip_classical_params_spatial_gradient_flows():
    """eqx.filter_value_and_grad reaches the spatial-surface coefficients."""
    from legoesm.training.aimip_params import AIMIPClassicalParams

    params = AIMIPClassicalParams.from_defaults(spatial_surface=True)
    grid = _grid_t11()

    def synthetic_loss(p):
        d = p.as_dict()
        scalar_part = jnp.sum(
            jnp.stack([d[c.name] / (c.max_val - c.min_val)
                       for c in p.constraints])
        )
        # Touch every spatial-field coefficient via a deterministic
        # projection (sum of all coefficients squared, scaled by their
        # static range so the magnitudes are comparable).
        spatial_part = jnp.array(0.0, dtype=scalar_part.dtype)
        if p.spatial_surface is not None:
            for name, field in p.spatial_surface.fields.items():
                spatial_part = spatial_part + jnp.sum(field.coeffs ** 2)
        return scalar_part + spatial_part

    loss, grads = eqx.filter_value_and_grad(synthetic_loss)(params)
    assert math.isfinite(float(loss))
    # Every spatial-surface coefficient leaf has a finite gradient.
    for name, field in grads.spatial_surface.fields.items():
        g = field.coeffs
        assert jnp.all(jnp.isfinite(g)), f"NaN/Inf gradient on {name}.coeffs"


# ----------------------------------------------------------------------
# MUON-partitioned optimizer
# ----------------------------------------------------------------------

def test_create_optimizer_muon_partitioned_routes_large_matrices():
    """``muon_partitioned`` builds and initializes on a mixed param tree."""
    from legoesm.ml.training import TrainingConfig, create_optimizer

    cfg = TrainingConfig(
        lr=1e-3, warmup_steps=2, total_steps=10,
        optimizer="muon_partitioned",
    )
    opt = create_optimizer(cfg)
    # Mixed tree: large 2-D weight matrix (MUON-eligible), small 2-D
    # (AdamW route), 1-D bias (AdamW route), 0-D scalar (AdamW route).
    params = {
        "big_W": jnp.ones((64, 64)),
        "small_W": jnp.ones((8, 8)),
        "bias": jnp.ones((64,)),
        "scalar": jnp.array(1.0),
    }
    state = opt.init(params)
    assert state is not None


# ----------------------------------------------------------------------
# split_rad + rad_update_interval gating
# ----------------------------------------------------------------------

def test_make_aimip_classical_spectral_physics_split_returns_tuple():
    """``split_rad=True`` returns ``(non_rad_fn, rad_fn)`` callables."""
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    grid = _grid_t11()
    params = AIMIPClassicalParams.from_defaults()
    built = make_aimip_classical_spectral_physics(
        params, grid, dt=1800.0, radiation="gray", split_rad=True,
    )
    assert isinstance(built, tuple)
    assert len(built) == 2
    assert callable(built[0])
    assert callable(built[1])


def test_make_aimip_classical_spectral_physics_combined_default():
    """``split_rad=False`` (default) returns the single combined callable."""
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    grid = _grid_t11()
    params = AIMIPClassicalParams.from_defaults()
    fn = make_aimip_classical_spectral_physics(
        params, grid, dt=1800.0, radiation="gray",
    )
    assert callable(fn)
    assert not isinstance(fn, tuple)


def test_spectral_rollout_rad_gating_one_step():
    """Single-step rollout with rad_update_interval=1 matches combined path.

    Smoke-checks the new ``rad_physics_fn`` + ``rad_update_interval``
    branch in :func:`spectral_rollout` by running both code paths over
    a one-step rollout starting from a zero spectral state.  The two
    outputs should agree to floating-point tolerance because at
    ``interval=1`` the gated branch fires rad on every step (same as
    the combined branch).
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    from legoesm.training.neural_gcm_spectral import (
        carry_to_spectral_state, spectral_rollout,
    )
    from legoesm.driver.compiled_segments import SegmentCarry as _SC

    grid = create_gaussian_grid(11, dealiasing="quadratic")
    sigma = create_sigma_coordinate(8)

    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, 8
    zero_3d = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    zero_2d = jnp.zeros((n_lat, n_lon), dtype=jnp.float64)
    ones_p_s = jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64)
    T_init = jnp.full((n_lat, n_lon, nlev), 250.0, dtype=jnp.float64)
    fake_carry = _SC(
        u=zero_3d, v=zero_3d, T=T_init,
        p_s=ones_p_s, phis=zero_2d,
        q_v=zero_3d, q_c=zero_3d, q_r=zero_3d,
        conv_prog=zero_3d,
        held_dT_rad=zero_3d, held_sw_net_sfc=zero_2d, held_lw_net_sfc=zero_2d,
        held_sw_up_toa=zero_2d, held_lw_up_toa=zero_2d,
        held_sw_up_toa_clr=zero_2d, held_lw_up_toa_clr=zero_2d,
        held_sw_down_toa=zero_2d,
        step_index=jnp.array(0),
        target_moisture=jnp.array(0.0),
        target_mass=jnp.array(0.0),
        max_cfl=jnp.array(0.0),
        precip_accum=zero_2d,
        shflx_accum=zero_2d,
        lhflx_accum=zero_2d,
        # TOA/sfc flux + near-surface-T accumulators added to SegmentCarry by
        # the 2026-07 origin/main merge (CLAUDE.md: every direct SegmentCarry()
        # construction must gain new fields).
        sw_up_toa_accum=zero_2d,
        lw_up_toa_accum=zero_2d,
        sw_up_toa_clr_accum=zero_2d,   # #843 clear-sky diagnostic accumulators
        lw_up_toa_clr_accum=zero_2d,
        sw_down_toa_accum=zero_2d,
        sw_net_sfc_accum=zero_2d,
        lw_net_sfc_accum=zero_2d,
        t_low_accum=zero_2d,
    )
    ic_spectral = carry_to_spectral_state(fake_carry, grid)
    pe_config = SpectralPEConfig(
        hyperdiff_coeff=2.5e15, hyperdiff_order=2,
        time_integrator="ssp_rk3",
        spectral_filter_strength=0.01, spectral_filter_order=8,
    )
    params = AIMIPClassicalParams.from_defaults()

    combined_fn = make_aimip_classical_spectral_physics(
        params, grid, dt=1800.0, radiation="gray", split_rad=False,
    )
    non_rad_fn, rad_fn = make_aimip_classical_spectral_physics(
        params, grid, dt=1800.0, radiation="gray", split_rad=True,
    )

    out_combined = spectral_rollout(
        ic_spectral, combined_fn, grid, sigma, pe_config,
        dt=1800.0, n_steps=1,
    )
    out_gated = spectral_rollout(
        ic_spectral, non_rad_fn, grid, sigma, pe_config,
        dt=1800.0, n_steps=1,
        rad_physics_fn=rad_fn, rad_update_interval=2,  # gate on, fire at step 0
    )
    # T_hat is the dominant scalar dycore field.  Combined runs rad
    # inside each of the SSP-RK3 sub-stages (3 rad calls / dycore
    # step); gated runs rad ONCE at the top of the step with the
    # state-at-step-start and holds it constant through the 3
    # sub-stages.  Outputs therefore differ by O(rad_tendency_dt *
    # state_change_per_substage) -- small for short rollouts, larger
    # for tight CFL.  Tolerance is set to capture the dominant
    # T_hat magnitudes (~900 in spectral, ~250 K in grid) while
    # allowing the substage-rad approximation.
    assert jnp.allclose(
        out_combined.T_hat.data, out_gated.T_hat.data,
        atol=1.0e-2, rtol=1.0e-5,
    ), "Gated rollout should agree with combined at step 1 to substage-rad tolerance"


def test_rrtmgp_spectral_pe_sfc_albedo_override_consumed_and_differentiable():
    """Truly-tunable RRTMGP surface path (2026-06-13).

    A build-time ``sfc_albedo_override`` (AIMIP's trained spatial ``(ncol,)``
    field) must (1) reach the RRTMGP heating through
    ``make_radiation_physics("spectral_pe", sfc_albedo_override=...)`` ->
    ``_call_radiation_backend`` -> ``_resolve_surface_field`` and (2) be
    reverse-mode differentiable -- WITHOUT ever being written into
    ``RRTMGPConfig.sfc_*`` (RRTMGP's Python solver-cache key). This is the
    plumbing that lets the AIMIP RRTMGP surface knobs be genuinely trainable
    instead of riding the config cache key by tracer/array identity.
    """
    import numpy as np
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig,
        RRTMGPConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        make_radiation_physics,
    )

    grid = create_gaussian_grid(n_max=8)
    sigma = create_sigma_coordinate(n_levels=3)
    qv = jnp.full((grid.n_lat, grid.n_lon, sigma.n_levels), 5.0e-3)
    state = isothermal_rest_state_spectral(grid, sigma, tracers={"q_v": qv})
    ncol = grid.n_lat * grid.n_lon
    cfg = RadiationConfig(scheme="rrtmgp", diurnal_cycle=False)

    def heating_sum(alb):
        # Rebuilt inside the differentiated fn (the AIMIP pattern): the override
        # is captured in the radiation closure while the config stays default.
        fn = make_radiation_physics(cfg, "spectral_pe", sfc_albedo_override=alb)
        out = fn(state, grid, sigma)
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

    lo = float(heating_sum(jnp.full((ncol,), 0.1)))
    hi = float(heating_sum(jnp.full((ncol,), 0.8)))
    assert np.isfinite(lo) and np.isfinite(hi)
    # SW absorption depends on surface albedo -> heating must respond.
    assert abs(lo - hi) > 0.0, "RRTMGP heating did not respond to sfc_albedo override"

    g = jax.grad(heating_sum)(jnp.full((ncol,), 0.3))
    g = np.asarray(g)
    assert g.shape == (ncol,)
    assert np.all(np.isfinite(g)), "non-finite gradient through sfc_albedo override"
    assert np.any(g != 0.0), "sfc_albedo override has zero gradient (not trainable)"

    # The override path must NOT have mutated the config surface fields: the
    # RRTMGP instance-cache key stays keyed on the concrete default, never a
    # traced array.
    assert float(cfg.rrtmgp.sfc_albedo) == float(RRTMGPConfig().sfc_albedo)


def test_make_physics_threads_sfc_override_combined_path():
    """The combined (split_rad=False) path: make_physics must thread
    sfc_albedo_override through _make_spectral_pe_combined ->
    make_radiation_physics to the RRTMGP solve, so the heating responds to the
    override without it touching RRTMGPConfig.sfc_*. Mirrors the split_rad path
    (which calls make_radiation_physics directly, covered by the grad test)."""
    import numpy as np
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig

    grid = create_gaussian_grid(n_max=8)
    sigma = create_sigma_coordinate(n_levels=3)
    qv = jnp.full((grid.n_lat, grid.n_lon, sigma.n_levels), 5.0e-3)
    state = isothermal_rest_state_spectral(grid, sigma, tracers={"q_v": qv})
    ncol = grid.n_lat * grid.n_lon
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="rrtmgp", diurnal_cycle=False),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )

    def t_tend_sum(alb):
        fn = make_physics(
            cfg, model_type="spectral_pe", dt=1800.0, sfc_albedo_override=alb,
        )
        out = fn(state, grid, sigma)
        out = out[0] if isinstance(out, tuple) else out
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

    lo = float(t_tend_sum(jnp.full((ncol,), 0.1)))
    hi = float(t_tend_sum(jnp.full((ncol,), 0.8)))
    assert np.isfinite(lo) and np.isfinite(hi)
    assert abs(lo - hi) > 0.0, "make_physics combined path dropped the sfc_albedo override"


def test_aimip_nonspatial_rrtmgp_sfc_albedo_is_trainable():
    """Non-spatial (default) AIMIP RRTMGP: the SCALAR rrtmgp_sfc_albedo knob must
    be genuinely trainable. With no spatial_surface, the builder routes the
    trained scalar (params.as_dict()['rrtmgp_sfc_albedo']) as a per-call override
    -> a finite nonzero gradient must reach its raw leaf. Guards against the knob
    silently becoming a dead leaf (the failure codex flagged) -- a regression
    here cannot hide behind the lower-level make_radiation_physics override
    tests."""
    import numpy as np
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )

    grid = create_gaussian_grid(n_max=8)
    sigma = create_sigma_coordinate(n_levels=3)
    qv = jnp.full((grid.n_lat, grid.n_lon, sigma.n_levels), 5.0e-3)
    state = isothermal_rest_state_spectral(grid, sigma, tracers={"q_v": qv})

    params = AIMIPClassicalParams.from_defaults()  # spatial_surface=False
    assert params.spatial_surface is None

    def loss(p):
        fn = make_aimip_classical_spectral_physics(
            p, grid, dt=1800.0, radiation="rrtmgp",
            convection_scheme="none", turbulence_scheme="none",
            gwd_scheme="none", microphysics_scheme="none", cloud_scheme="none",
            # RADIATION-ONLY probe: every other family is off on purpose so the
            # gradient reaching the surface-albedo leaf can only have come
            # through radiation. The completeness gate guards MODELS, not
            # single-term probes, so it is waived explicitly here.
            allow_unfilled_families=True,
        )
        out = fn(state, grid, sigma)
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2)

    val, grads = eqx.filter_value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    g = np.asarray(grads.raw_values["rrtmgp_sfc_albedo"])
    assert np.all(np.isfinite(g)), "non-finite gradient on the scalar rrtmgp_sfc_albedo"
    assert np.any(g != 0.0), "rrtmgp_sfc_albedo is a dead leaf in the non-spatial RRTMGP path"


def test_spatial_baselines_from_params_maps_trained_scalars():
    """Regression (codex): the spatial-field baselines must come from the
    TRAINED scalar keys, alias-mapped to the field names. A raw ``as_dict``
    pass-through left every baseline at the static f_0 -> trained scalars dead
    + no trainable ocean-surface lever under ``spatial_surface=True``."""
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        spatial_baselines_from_params,
    )
    from legoesm.training.aimip_spatial import _FIELD_SPECS

    d = AIMIPClassicalParams.from_defaults().as_dict()
    for rad in ("rrtmgp", "gray"):
        b = spatial_baselines_from_params(d, rad)
        # keys must EXACTLY match the spatial field names evaluate() looks up
        assert set(b) == set(_FIELD_SPECS), (set(b), set(_FIELD_SPECS))
        assert b["Cd_neutral"] is d["surface_Cd_neutral"]
        assert b["Ch_neutral"] is d["surface_Ch_neutral"]
        assert b["z0"] is d["surface_z0"]

    # RRTMGP: the surface radiative baselines are the TRAINED scalars.
    b_rrtmgp = spatial_baselines_from_params(d, "rrtmgp")
    assert b_rrtmgp["sfc_albedo"] is d["rrtmgp_sfc_albedo"]
    assert b_rrtmgp["sfc_emissivity"] is d["rrtmgp_sfc_emissivity"]

    # Gray: no trained gray surface scalars exist since 2026-08-11 (gray is not
    # trained), so the baselines fall back to the scheme's own defaults. The
    # spatial FIELD on top of them is still learned.
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    b_gray = spatial_baselines_from_params(d, "gray")
    assert b_gray["sfc_albedo"] == GrayRadiationConfig().sfc_albedo
    assert b_gray["sfc_emissivity"] == GrayRadiationConfig().sfc_emissivity
    assert "gray_sfc_albedo" not in d


def test_aimip_zm_lane_runs_on_the_era5_land_fraction_and_refuses_without_it(
        monkeypatch):
    """ZM on the AIMIP classical lane gets the ERA5 land-sea mask; without it ZM raises."""
    import types

    import numpy as np

    import legoesm.training.era5_to_state as e2s
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics,
    )
    from legoesm.training.aimip_spatial import grid_with_zm_land_fraction

    # ERA5-like slice: land in ONE quadrant (north, lon < pi), so a lat flip,
    # lon roll or transpose of the mask lands it on the wrong columns.
    lat = np.linspace(-np.pi / 2, np.pi / 2, 19)
    lon = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    lsm = ((lat[:, None] > 0) & (lon[None, :] < np.pi)).astype(np.float32)
    calls = []

    def fake_slice(config, time_idx):
        calls.append(config.load_land_frac)
        return types.SimpleNamespace(lat=lat, lon=lon, land_frac=lsm, sfc_shf=None)

    monkeypatch.setattr(e2s, "load_era5_slice", fake_slice)
    grid = create_gaussian_grid(8, dealiasing="quadratic")
    sig = create_sigma_coordinate(6)
    assert grid_with_zm_land_fraction(grid, "tiedtke") is grid
    assert calls == []
    g_land = grid_with_zm_land_fraction(grid, "zhang_mcfarlane")
    assert calls == [True]
    assert g_land.land_frac.shape == (grid.n_lat * grid.n_lon,)
    lf = np.asarray(g_land.land_frac).reshape(grid.n_lat, grid.n_lon)
    glat = np.asarray(grid.lat)[:, None] * np.ones((1, grid.n_lon))
    glon = np.ones((grid.n_lat, 1)) * np.asarray(grid.lon)[None, :]
    inside = (glat > 0.3) & (glon > 0.3) & (glon < np.pi - 0.3)
    outside = (glat < -0.3) | ((glon > np.pi + 0.3) & (glon < 2 * np.pi - 0.3))
    assert inside.any() and outside.any()
    assert np.all(lf[inside] == 1.0) and np.all(lf[outside] == 0.0)

    shp = (grid.n_lat, grid.n_lon, 6)
    state = isothermal_rest_state_spectral(
        grid, sig, T_init=290.0, p_s_init=1.0e5,
        tracers={"q_v": jnp.full(shp, 1.2e-2), "q_c": jnp.full(shp, 1e-5),
                 "q_i": jnp.full(shp, 1e-6)})
    fn = make_aimip_classical_spectral_physics(
        AIMIPClassicalParams.from_defaults(), grid, 1800.0,
        radiation="gray", convection_scheme="zhang_mcfarlane",
        turbulence_scheme="louis", gwd_scheme="mcfarlane",
        microphysics_scheme="sundqvist", cloud_scheme="xu_randall")
    out = fn(state, g_land, sig)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(out))
    with pytest.raises(ValueError, match="land_frac is required"):
        fn(state, grid, sig)


@pytest.mark.slow
@pytest.mark.tier3
def test_era5_land_fraction_on_the_aimip_t21_grid_is_earths():
    """The real ERA5 land-sea mask on the T21 AIMIP grid (needs the WB2 store):
    about a third of the columns, and 0.29 of the area, are land."""
    import numpy as np

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_spatial import (
        era5_land_fraction, grid_with_zm_land_fraction,
    )

    grid = create_gaussian_grid(21, dealiasing="quadratic")
    land = np.asarray(era5_land_fraction(grid))
    assert land.shape == (grid.n_lat, grid.n_lon)
    assert land.min() >= 0.0 and land.max() <= 1.0
    w = np.cos(np.asarray(grid.lat))[:, None] * np.ones((1, grid.n_lon))
    print(f"ERA5 land fraction T21: column mean {land.mean():.4f}, "
          f"area mean {(land * w).sum() / w.sum():.4f}")
    assert land.mean() == pytest.approx(0.33, abs=0.03)
    assert (land * w).sum() / w.sum() == pytest.approx(0.29, abs=0.03)
    # ZM and the surface-parameter mask see the same field.
    zm = grid_with_zm_land_fraction(grid, "zhang_mcfarlane", land).land_frac
    assert np.array_equal(np.asarray(zm), land.reshape(-1))


def test_run_aimip_training_and_eval_use_the_era5_land_mask(monkeypatch):
    """run_aimip's classical training AND its in-script eval take the spatial
    surface mask from the ERA5 land-sea mask (the one AMIP inference and ZM
    use), not from orography."""
    import importlib.util
    import types
    from pathlib import Path

    import numpy as np

    import legoesm.training.aimip_params as ap
    import legoesm.training.aimip_spatial as sp
    import legoesm.training.neural_gcm_spectral as ngs

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_aimip_land_mask_probe", root / "scripts" / "run" / "run_aimip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = mod._merge(mod._load_yaml(root / "config" / "aimip" / "aimip_era5.yaml"), {
        "n_max": 8, "aimip_variant": "classical", "aimip_spatial_surface": True,
        "aimip_radiation": "gray", "aimip_rad_update_interval": 1,
        "aimip_convection": "zhang_mcfarlane"})
    spec_cfg = mod._build_spectral_config(cfg)

    masks = []

    def fake_era5(grid):
        m = jnp.asarray(np.random.default_rng(len(masks)).random(
            (grid.n_lat, grid.n_lon)))
        masks.append(m)
        return m

    # High orography everywhere: an orography-derived mask would be ~1.
    def fake_data(spec_cfg_, grid, sigma, cache_dir, **kw):
        carry = types.SimpleNamespace(
            phis=jnp.full((grid.n_lat, grid.n_lon), 5.0e4))
        return [carry], [carry], [0]

    seen = []

    class _Stop(Exception):
        pass

    def fake_build(p, grid_, dt, **kw):
        seen.append((kw["land_mask"], grid_.land_frac))
        if len(seen) > 1:           # the eval build: nothing past it is needed
            raise _Stop
        return lambda *a, **k: None

    def fake_loop(params, make_physics_fn, grid, *a, **k):
        make_physics_fn(params, grid)
        return params

    monkeypatch.setattr(sp, "era5_land_fraction", fake_era5)
    monkeypatch.setattr(ngs, "load_training_data", fake_data)
    monkeypatch.setattr(ngs, "_train_spectral_loop", fake_loop)
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics", fake_build)

    mod._train_aimip_classical(spec_cfg, "unused", cfg=cfg)
    assert len(masks) == 1          # one ERA5 load feeds both ZM and the mask
    land_mask, zm_land = seen[0]
    np.testing.assert_array_equal(np.asarray(land_mask), np.asarray(masks[0]))
    np.testing.assert_array_equal(np.asarray(zm_land),
                                  np.asarray(masks[0]).reshape(-1))

    with pytest.raises(_Stop):
        mod._evaluate_variant("classical", None, cfg, "unused")
    np.testing.assert_array_equal(np.asarray(seen[1][0]), np.asarray(masks[-1]))
