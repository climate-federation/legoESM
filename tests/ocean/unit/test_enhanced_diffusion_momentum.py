"""Direct unit tests for the convective **momentum viscosity** of the
``enhanced_diffusion`` scheme (Oceananigans ``convective_νz`` parity).

The enhanced-diffusion convective adjustment previously mixed only
tracers; momentum got the tracer ``K_conv`` reused implicitly inside
``compute_vertical_K_profiles``.  These tests guard the now-independent
``nu_conv`` / ``nu_bg`` viscosity:

  * ``A_v`` is built from ``nu_conv`` where ``N² < 0`` and ``nu_bg``
    where stable — independent of the tracer ``K_v``.
  * the explicit branch mixes momentum (reduces shear) where unstable.
  * ``nu_conv = 0`` disables momentum mixing while tracers still mix
    (recovers Oceananigans' default ``convective_νz = 0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig, OceanConvectionConfig,
)
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    enhanced_diffusion_convection,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _build_unstable_column(n_levels: int = 8, H_max: float = 4000.0):
    """6-face cubed-sphere column with a statically unstable top.

    Returns T, S, rho (linear EOS consistent with T), u, v, z_coord, J.
    Surface layer is made cold/dense so ``N² < 0`` at the top interface.
    """
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max)
    shape = (6, 4, 4, n_levels)

    # Stable below, dense (cold) cap on top → N² < 0 at the first interface.
    T_profile = jnp.linspace(18.0, 2.0, n_levels)
    T = jnp.broadcast_to(T_profile, shape).astype(jnp.float64)
    T = T.at[..., 0].set(0.0)  # cold dense surface

    S = jnp.full(shape, 35.0, dtype=jnp.float64)
    # Linear EOS: rho decreases with T (alpha>0) → cold top is denser.
    rho = constants.rho_ocean - 0.2 * (T - 4.0)

    # Sheared velocity so vertical momentum diffusion has work to do.
    u_profile = jnp.linspace(0.5, -0.5, n_levels)
    u = jnp.broadcast_to(u_profile, shape).astype(jnp.float64)
    v = jnp.zeros(shape, dtype=jnp.float64)

    J = jnp.ones((6, 4, 4), dtype=jnp.float64)
    return T, S, rho, u, v, z, J


def test_A_v_independent_of_K_v():
    """A_v uses nu_conv/nu_bg, K_v uses K_conv/K_bg — distinct fields."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    cfg = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5,
        nu_conv=0.3, nu_bg=2e-5,           # deliberately != tracer values
        smooth_transition=False,           # hard step → exact comparison
    )
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=False, u=u, v=v,
    )
    assert out.K_v is not None and out.A_v is not None

    # Where unstable (N²<0) K_v→K_conv, A_v→nu_conv; elsewhere backgrounds.
    unstable = out.convection_flag > 0.5
    # Top interface must be unstable given the cold cap.
    assert bool(jnp.any(unstable))

    K_on = jnp.where(unstable, out.K_v, cfg.K_conv)
    A_on = jnp.where(unstable, out.A_v, cfg.nu_conv)
    assert jnp.allclose(K_on[unstable], cfg.K_conv)
    assert jnp.allclose(A_on[unstable], cfg.nu_conv)

    stable = ~unstable
    if bool(jnp.any(stable)):
        assert jnp.allclose(out.K_v[stable], cfg.K_bg)
        assert jnp.allclose(out.A_v[stable], cfg.nu_bg)


def test_explicit_momentum_mixing_reduces_shear():
    """Explicit branch with nu_conv>0 produces a shear-reducing du/dt."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    cfg = EnhancedDiffusionConfig(nu_conv=1.0, nu_bg=1e-5)
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=True, u=u, v=v, dt=600.0,
    )
    assert out.du_dt is not None
    assert jnp.all(jnp.isfinite(out.du_dt))
    # Momentum is mixed: tendency is non-trivial somewhere in the column.
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0

    # Diffusion damps shear: the top cell (u=+0.5) sees du/dt<0, pulling it
    # toward the layer below (smaller u). The zero-flux operator conserves
    # the *volume-integrated* momentum ∑(du/dt · dz) (not the plain sum,
    # since z-star layers differ in thickness) → ~0 per column.
    dz = z.dz_ref * J[..., jnp.newaxis]              # (6,4,4,nlev), J=1 here
    col_mom = jnp.sum(out.du_dt * dz, axis=-1)
    assert jnp.allclose(col_mom, 0.0, atol=1e-6)
    assert float(out.du_dt[0, 0, 0, 0]) < 0.0


def test_nu_conv_zero_disables_momentum_mixing():
    """nu_conv=nu_bg=0 → no momentum mixing; tracers still convect."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    cfg = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5, nu_conv=0.0, nu_bg=0.0,
    )
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=True, u=u, v=v, dt=600.0,
    )
    # Momentum untouched.
    assert float(jnp.max(jnp.abs(out.du_dt))) == 0.0
    assert float(jnp.max(jnp.abs(out.dv_dt))) == 0.0
    assert jnp.allclose(out.A_v, 0.0)
    # Tracers still mixed by the convective adjustment.
    assert float(jnp.max(jnp.abs(out.dT_dt))) > 0.0


def test_dry_column_A_v_finite_and_zero():
    """Dry columns (jacobian=0) must yield finite, zero A_v/K_v.

    ``compute_buoyancy_frequency`` divides by dz_interface (=0 on dry
    columns).  The scheme substitutes safe inputs BEFORE that division so
    both forward value and reverse-mode gradient stay finite, then zeros
    K_v / A_v on dry interfaces.
    """
    T, S, rho, u, v, z, J = _build_unstable_column()
    # Mark face 0 fully dry (jacobian 0). rho stays finite & T-dependent —
    # the realistic dry-cell case that exercises the 1/dz division in AD.
    J = J.at[0].set(0.0)
    cfg = EnhancedDiffusionConfig(nu_conv=0.5, nu_bg=2e-5)

    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=False, u=u, v=v,
    )
    assert jnp.all(jnp.isfinite(out.K_v))
    assert jnp.all(jnp.isfinite(out.A_v))
    # Dry face → zero mixing.
    assert jnp.all(out.K_v[0] == 0.0)
    assert jnp.all(out.A_v[0] == 0.0)


def test_dry_column_gradient_finite():
    """Reverse-mode AD through a dry column (J=0, finite T-dependent rho).

    Guards the Codex finding that masking N² *after* the 1/dz division
    leaves a ``0·∞ = NaN`` cotangent.  rho depends on T, so the gradient
    path runs through the (safe-substituted) buoyancy-frequency division.
    """
    T, S, rho0, u, v, z, J = _build_unstable_column()
    J = J.at[0].set(0.0)
    cfg = EnhancedDiffusionConfig(nu_conv=0.5, nu_bg=2e-5)

    def _loss(T_):
        rho_ = constants.rho_ocean - 0.2 * (T_ - 4.0)   # finite, depends on T
        o = enhanced_diffusion_convection(
            T_, S, rho_, z, J, cfg, apply_diffusion=False, u=u, v=v,
        )
        return jnp.sum(o.A_v) + jnp.sum(o.K_v)

    g = jax.grad(_loss)(T)
    assert jnp.all(jnp.isfinite(g)), "NaN gradient through dry-column N² division"


def test_dry_column_nan_density_forward_finite():
    """Even a NaN density sentinel on a dry column must not leak forward."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    J = J.at[0].set(0.0)
    rho = rho.at[0].set(jnp.nan)
    cfg = EnhancedDiffusionConfig(nu_conv=0.5, nu_bg=2e-5)
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=False, u=u, v=v,
    )
    assert jnp.all(jnp.isfinite(out.K_v))
    assert jnp.all(jnp.isfinite(out.A_v))
    assert jnp.all(out.A_v[0] == 0.0)


def test_velocity_optional_back_compat():
    """Omitting u/v with nu=0 (default): tracer branch runs, du/dv None."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    cfg = EnhancedDiffusionConfig()                      # nu_conv=nu_bg=0
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=True,
    )
    assert out.du_dt is None and out.dv_dt is None
    assert out.A_v is not None                # viscosity field still reported
    assert float(jnp.max(jnp.abs(out.dT_dt))) > 0.0


def test_kernel_explicit_nu_conv_without_velocity_raises():
    """Kernel fail-closed: apply_diffusion=True + nu_conv>0 + omitted u/v
    must raise rather than silently take the tracer-only branch."""
    T, S, rho, u, v, z, J = _build_unstable_column()
    cfg = EnhancedDiffusionConfig(nu_conv=1.0)
    with pytest.raises(ValueError, match="u/v were not supplied"):
        enhanced_diffusion_convection(
            T, S, rho, z, J, cfg, apply_diffusion=True,   # no u/v
        )
    # nu=0 → tracer-only legal (no raise); implicit mode also legal.
    cfg0 = EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0)
    enhanced_diffusion_convection(T, S, rho, z, J, cfg0, apply_diffusion=True)
    enhanced_diffusion_convection(T, S, rho, z, J, cfg, apply_diffusion=False)


def _latlon_state_and_configs():
    """Lat-lon C-grid state + (KPP+conv) / (KPP-only) physics configs."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, KPPConfig,
    )

    grid = create_latlon_grid(n_lat=12, n_lon=24)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    # Statically unstable surface cap so convection would fire.
    T = jnp.asarray(state.T.data)
    T = T.at[..., 0].set(1.0)
    state = state._replace(T=state.T.replace(data=T))

    base = dict(
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        shortwave_penetration=None,
    )
    kpp = VerticalMixingConfig(scheme="kpp", kpp=KPPConfig())
    # Tracer-only convection (nu=0) — the only convection config KPP accepts.
    conv = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0),
    )
    cfg_kpp_conv = OceanPhysicsConfig(
        vertical_mixing=kpp, convection=conv, **base,
    )
    cfg_kpp_only = OceanPhysicsConfig(
        vertical_mixing=kpp,
        convection=OceanConvectionConfig(scheme="none"),
        **base,
    )
    return grid, z, state, cfg_kpp_conv, cfg_kpp_only


def test_kpp_plus_nu_conv_rejected():
    """KPP + enhanced_diffusion(nu_conv>0) must raise — KPP owns interior
    convective momentum, so the user's nu_* cannot be honoured and must not
    be silently suppressed."""
    from legoesm.ocean.physics.combined import (
        OceanPhysicsConfig, make_ocean_physics,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, KPPConfig,
    )

    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=KPPConfig()),
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=5.0),
        ),
        shortwave_penetration=None,
    )
    with pytest.raises(ValueError, match="cannot be combined with KPP"):
        make_ocean_physics(cfg, apply_vertical_diffusion=False)


def test_kpp_plus_tracer_only_convection_ok():
    """KPP + enhanced_diffusion(nu=0): accepted.  A_v equals KPP-only (no
    convective momentum added) while K_v still gets the convective tracer
    boost where the column is unstable."""
    from legoesm.ocean.physics.combined import make_ocean_physics

    grid, z, state, cfg_kpp_conv, cfg_kpp_only = _latlon_state_and_configs()

    fn_conv = make_ocean_physics(cfg_kpp_conv, apply_vertical_diffusion=False)
    fn_only = make_ocean_physics(cfg_kpp_only, apply_vertical_diffusion=False)
    t_conv = fn_conv(state, grid, z, None)
    t_only = fn_only(state, grid, z, None)

    assert t_conv.A_v is not None and t_only.A_v is not None
    assert jnp.allclose(t_conv.A_v, t_only.A_v), \
        "tracer-only convection must not change momentum A_v under KPP"
    # Tracer: convection DOES boost K_v (unstable surface), so they differ.
    assert not jnp.allclose(t_conv.K_v, t_only.K_v)


def test_cgrid_explicit_nu_conv_rejected():
    """C-grid + explicit mode + nu_conv>0 must raise (not silently drop).

    Staggered velocities cannot use the cell-centred explicit momentum
    operator, and explicit mode has no implicit A_v solve to apply the
    convective viscosity to the faces — so the configured momentum mixing
    would be silently lost.  The factory must fail loudly instead.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics,
    )

    grid = create_latlon_grid(n_lat=12, n_lon=24)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )

    cfg = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=1.0),
    )
    fn = make_convection_physics(
        cfg, apply_diffusion=True, emit_momentum_viscosity=True,
    )
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        fn(state, grid, z, None)

    # nu_conv = nu_bg = 0 → tracer-only, no error on the same C-grid path.
    cfg0 = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0),
    )
    fn0 = make_convection_physics(
        cfg0, apply_diffusion=True, emit_momentum_viscosity=True,
    )
    out = fn0(state, grid, z, None)
    assert float(jnp.max(jnp.abs(out.dT_dt.data))) >= 0.0  # runs, no raise


def test_direct_factory_suppression_with_nu_conv_rejected():
    """Direct make_convection_physics(emit_momentum_viscosity=False) with a
    nonzero nu_conv/nu_bg must raise at construction — suppression is only
    valid for tracer-only (nu=0)."""
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics,
    )

    cfg = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=1.0),
    )
    with pytest.raises(ValueError, match="emit_momentum_viscosity=False"):
        make_convection_physics(
            cfg, apply_diffusion=False, emit_momentum_viscosity=False,
        )
    # nu=0 suppression is legal (KPP tracer-only path).
    cfg0 = OceanConvectionConfig(
        scheme="enhanced_diffusion",
        enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0),
    )
    assert make_convection_physics(
        cfg0, apply_diffusion=False, emit_momentum_viscosity=False,
    ) is not None


def test_cgrid_explicit_kpp_nu_conv_rejected_at_construction():
    """make_ocean_physics with KPP + enhanced_diffusion(nu_conv>0) must raise
    at construction (KPP guard), regardless of explicit/implicit mode —
    the KPP composition cannot silently swallow the configured nu_conv."""
    from legoesm.ocean.physics.combined import (
        OceanPhysicsConfig, make_ocean_physics,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, KPPConfig,
    )

    cfg = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=KPPConfig()),
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=1.0),
        ),
        shortwave_penetration=None,
    )
    # Both modes rejected at construction by the KPP guard.
    with pytest.raises(ValueError, match="cannot be combined with KPP"):
        make_ocean_physics(cfg, apply_vertical_diffusion=True)
    with pytest.raises(ValueError, match="cannot be combined with KPP"):
        make_ocean_physics(cfg, apply_vertical_diffusion=False)


def test_fallback_adds_convective_viscosity_when_not_kpp():
    """The implicit fallback builder applies the convective momentum
    viscosity (nu_conv) when vmix != "kpp" (here: constant).

    Complements ``test_kpp_composition_no_A_v_double_count`` (which checks
    the suppression under KPP) and the constant+convection explicit/
    implicit equivalence in ``test_implicit_vertical_mixing`` (which
    checks fast-path == fallback for the constant scheme).
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig, ConstantVerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )

    grid = create_latlon_grid(n_lat=12, n_lon=24)
    z = create_ocean_z_star(n_levels=6, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    T = jnp.asarray(state.T.data).at[..., 0].set(1.0)   # unstable cap
    state = state._replace(T=state.T.replace(data=T))

    def _cfg(nu):
        return OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
            surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(nu_conv=nu, nu_bg=1e-5),
            ),
            shortwave_penetration=None,
        )

    _, A_hi = compute_vertical_K_profiles(state, z, None, _cfg(5.0))
    _, A_lo = compute_vertical_K_profiles(state, z, None, _cfg(0.0))
    # nu_conv raises A_v at the unstable surface interface; nu_conv=0 does not.
    assert float(jnp.max(A_hi)) > 1.0
    assert float(jnp.max(A_lo)) < 1e-2
    assert float(jnp.max(A_hi - A_lo)) > 1.0
