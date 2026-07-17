"""Category 7: Ocean Physics -- Smoke Tests & Physical Consistency.

Tests vertical mixing, bottom drag, and ocean convection for finite
outputs, physical sign conventions, and stability responses.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init import rest_state_ocean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ocean_state(n=8, nlev=10):
    """Create a small ocean state for testing."""
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    state = rest_state_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0,
    )
    # Add small currents
    state = state._replace(
        u=Field(data=jnp.ones_like(state.u.data) * 0.1,
                name="u", dims=state.u.dims, units="m/s"),
        v=Field(data=jnp.ones_like(state.v.data) * 0.05,
                name="v", dims=state.v.dims, units="m/s"),
    )
    return state, grid, z_coord


# ============================================================================
# 7a  Vertical mixing smoke tests
# ============================================================================

def test_richardson_prandtl_grows_with_ri():
    """Pacanowski & Philander (1981) PP81 mandates that the Prandtl
    ratio ν/κ = 1 + α·Ri GROWS with Richardson number, because in stable
    shear momentum mixes more efficiently than tracer.

    This test exercises the *full pipeline* — calling
    ``richardson_vertical_mixing`` on a low-Ri and a high-Ri column —
    and asserts that the returned A_v / K_v ratio is materially larger
    in the high-Ri column.

    Construction:
    - low-Ri: shear-dominated (large du/dz, weak N²): Ri ≈ 0
    - high-Ri: stable-stratification-dominated (weak shear, strong N²)

    Why non-vacuous: under the prior bug (A_v = K_v · constant_Pr_t),
    the ratio A_v/K_v == 10 in both columns regardless of Ri — this
    assertion would fail by construction.
    """
    from legoesm.ocean.physics.vertical_mixing.richardson import (
        richardson_vertical_mixing,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        RichardsonVerticalMixingConfig,
    )

    state, grid, z_coord = _make_ocean_state()
    cfg = RichardsonVerticalMixingConfig()
    jacobian = jnp.ones_like(state.eta.data)

    # Low-Ri column: large velocity gradient, weak T-stratification
    u_shear = jnp.zeros_like(state.u.data).at[..., :].set(
        jnp.linspace(0.0, 1.0, state.u.data.shape[-1])
    )
    T_weak = jnp.full_like(state.T.data, 10.0)  # quasi-isothermal
    S_uniform = jnp.full_like(state.S.data, 35.0)
    # rho weak gradient — match T quasi-isothermal
    rho_weak = jnp.full_like(state.T.data, 1027.0)

    out_low_ri = richardson_vertical_mixing(
        u_shear, jnp.zeros_like(state.v.data),
        T_weak, S_uniform, rho_weak, z_coord, jacobian, cfg,
    )

    # High-Ri column: zero velocity gradient, strong T-stratification
    u_quiet = jnp.zeros_like(state.u.data)
    nlev = state.T.data.shape[-1]
    # Strong stratification: T decreasing with depth (warm at top)
    T_stratified = state.T.data  # already has T_water_init_C=20, T_deep=2
    rho_strong = jnp.broadcast_to(
        jnp.linspace(1024.0, 1030.0, nlev)[None, None, None, :],
        state.T.data.shape,
    )

    out_high_ri = richardson_vertical_mixing(
        u_quiet, jnp.zeros_like(state.v.data),
        T_stratified, S_uniform, rho_strong, z_coord, jacobian, cfg,
    )

    # PP81: Pr = A_v / K_v grows with Ri.  Floor by background to avoid
    # zero-division when both A_v and K_v are background-dominated.
    Pr_low = float(jnp.mean(out_low_ri.A_v / jnp.maximum(out_low_ri.K_v, 1e-10)))
    Pr_high = float(jnp.mean(out_high_ri.A_v / jnp.maximum(out_high_ri.K_v, 1e-10)))

    assert Pr_high > Pr_low + 0.1, (
        f"Pacanowski–Philander 1981 Prandtl-Ri scaling broken: "
        f"Pr(low Ri) = {Pr_low:.3f}, Pr(high Ri) = {Pr_high:.3f}. "
        f"Under the prior bug both ratios would equal Pr_t = 10."
    )


def test_kpp_unstable_prandtl_below_one():
    """LMD94: in a convectively-unstable boundary layer the turbulent Prandtl
    number Pr_t = A_v/K_v < 1 — scalars mix MORE efficiently than momentum
    (scalar velocity scale w_s exceeds the momentum scale w_m).  Executable spec
    for the F-OCEAN-1 fix.

    Non-vacuous: the current single-scale scheme gives A_v ≳ K_v in the BL (they
    differ only by the background floors A_bg=1e-4 > K_bg=1e-5 ⇒ Pr_t ≈ 1.01,
    verified) — so this assertion fails until ``w_s > w_m`` is implemented and the
    tracer BL diffusivity overtakes the momentum viscosity.
    """
    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    from legoesm.ocean.physics.vertical_mixing.config import KPPConfig

    n, nlev = 8, 12
    grid = create_cubed_sphere(n)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=2000.0)
    state = rest_state_ocean(
        grid, z_coord, T_water_init_C=18.0, T_deep=4.0, S_uniform=35.0,
        H_max=2000.0,
    )
    sh = state.T.data.shape
    rho = jnp.broadcast_to(
        jnp.linspace(1025.0, 1028.0, nlev)[None, None, None, :], sh,
    )  # stable background stratification
    surf = state.eta.data.shape
    B_f = jnp.full(surf, 5e-7)   # strong destabilizing (convective) surface buoyancy flux
    tau_x = jnp.full(surf, 0.1)  # wind stress -> nonzero u_star
    tau_y = jnp.zeros(surf)

    out = kpp_vertical_mixing(
        state.u.data, state.v.data, state.T.data, state.S.data, rho,
        state.eta.data, z_coord, jnp.ones(surf), KPPConfig(),
        tau_x=tau_x, tau_y=tau_y, B_f=B_f, apply_diffusion=False,
    )
    # Top-3 interfaces sit inside the convective boundary layer.
    K_v_bl = float(jnp.mean(out.K_v[..., :3]))
    A_v_bl = float(jnp.mean(out.A_v[..., :3]))
    assert K_v_bl > A_v_bl, (
        f"Unstable-BL Pr_t = A_v/K_v = {A_v_bl / K_v_bl:.3f} ≥ 1; LMD94 requires "
        f"Pr_t < 1 (tracer must mix more than momentum)."
    )


@pytest.mark.parametrize("scheme", ["constant", "richardson", "kpp"])
def test_vertical_mixing_smoke(scheme):
    """Each vertical mixing scheme produces finite outputs."""
    from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig

    state, grid, z_coord = _make_ocean_state()
    config = VerticalMixingConfig(scheme=scheme)
    phys_fn = make_vertical_mixing_physics(config)
    tend = phys_fn(state, grid, z_coord)

    assert jnp.all(jnp.isfinite(tend.du_dt.data)), f"{scheme}: du_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dv_dt.data)), f"{scheme}: dv_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt has NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt has NaN/Inf"


# ============================================================================
# 7g  Bottom drag smoke tests and sign checks
# ============================================================================

@pytest.mark.parametrize("scheme", ["enhanced_diffusion", "plume"])
def test_ocean_convection_smoke(scheme):
    """Each ocean convection scheme produces finite outputs."""
    from legoesm.ocean.physics.convection.integration import (
        make_convection_physics as make_ocean_convection,
    )
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig

    state, grid, z_coord = _make_ocean_state()
    config = OceanConvectionConfig(scheme=scheme)
    phys_fn = make_ocean_convection(config)
    tend = phys_fn(state, grid, z_coord)

    assert jnp.all(jnp.isfinite(tend.dT_dt.data)), f"{scheme}: dT_dt NaN/Inf"
    assert jnp.all(jnp.isfinite(tend.dS_dt.data)), f"{scheme}: dS_dt NaN/Inf"
