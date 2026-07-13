"""Regression: the pipeline convection module must NOT emit K under
fallback-computed vmix schemes (tke/catke) in implicit mode.

TKE/CATKE pipeline factories are deliberate no-ops (K_v=None) so the
model's implicit solve takes the compute_vertical_K_profiles fallback,
which composes the closure K WITH enhanced-diffusion convection.  When
the pipeline's convection module also emitted K, the summed K_v/A_v was
non-None and _apply_implicit_vertical_mixing silently took the
physics-provided-K FAST path, skipping the TKE closure entirely (DINO
r1_exact equatorial-jet blowup, probe job 8818083).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.physics.combined import (
    OceanPhysicsConfig, make_ocean_physics,
)
from legoesm.ocean.physics.convection.config import (
    EnhancedDiffusionConfig, OceanConvectionConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    TKEConfig, VerticalMixingConfig,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star


def _mk_state_grid_z(nlev=6, H_max=2000.0):
    grid = create_latlon_grid(8, 12)
    z = create_ocean_z_star(nlev, H_max=H_max)
    state = rest_state_latlon_cgrid_ocean(grid, z, H_max=H_max)
    # cell-centred velocity view like the model's cc_state
    u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
    v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
    state = state._replace(u=state.u.replace(data=u_cell),
                           v=state.v.replace(data=v_cell))
    return state, grid, z


def _physics_cfg(vmix_scheme):
    if vmix_scheme == "tke":
        vm = VerticalMixingConfig(scheme="tke", tke=TKEConfig())
    else:
        vm = VerticalMixingConfig(scheme=vmix_scheme)
    return OceanPhysicsConfig(
        vertical_mixing=vm,
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        shortwave_penetration=None,
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=10.0, nu_conv=10.0),
        ),
    )


def test_tke_plus_evd_pipeline_emits_no_K():
    """Under tke + enhanced_diffusion (implicit mode) the pipeline output
    K_v/A_v must be None so the model takes the TKE fallback."""
    state, grid, z = _mk_state_grid_z()
    fn = make_ocean_physics(_physics_cfg("tke"),
                            apply_vertical_diffusion=False)
    tend = fn(state, grid, z, None)
    assert tend.K_v is None
    assert tend.A_v is None


def test_constant_plus_evd_pipeline_still_emits_K():
    """Non-fallback schemes keep the fast path: constant + EVD sums K."""
    state, grid, z = _mk_state_grid_z()
    fn = make_ocean_physics(_physics_cfg("constant"),
                            apply_vertical_diffusion=False)
    tend = fn(state, grid, z, None)
    assert tend.K_v is not None
    assert tend.A_v is not None
    assert float(jnp.max(tend.K_v)) >= 0.0


def test_fallback_composes_tke_with_evd():
    """compute_vertical_K_profiles under tke + EVD returns profiles that
    include the convective enhancement where a column is unstable."""
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )
    # shallow column so adiabatic compression cannot mask the inversion
    state, grid, z = _mk_state_grid_z(H_max=200.0)
    # statically unstable column: warm below, cold above
    T = np.broadcast_to(
        np.linspace(5.0, 15.0, state.T.data.shape[-1])[None, None, :],
        state.T.data.shape).copy()
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    K_v, A_v = compute_vertical_K_profiles(
        state, z, None, _physics_cfg("tke"),
        A_v_background=1e-4, K_v_background=1e-5)
    # EVD K_conv=10 must dominate the interior of the unstable columns
    assert float(jnp.max(K_v)) >= 10.0
    assert float(jnp.max(A_v)) >= 10.0


def test_scm_rejects_fallback_computed_schemes():
    """The ocean SCM has no compute_vertical_K_profiles fallback, so the
    fallback-computed closures (tke/catke) must be rejected there rather
    than silently running with background-only mixing."""
    from legoesm.ocean.scm import OceanColumnModel

    with pytest.raises(NotImplementedError, match="tke"):
        OceanColumnModel.create(
            nlev=8, dt=1800.0,
            T_profile=jnp.linspace(18.0, 4.0, 8),
            physics_config=_physics_cfg("tke"),
            implicit_vertical_mixing=True,
        )
