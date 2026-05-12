"""Integration tests for ``LatLonCGridOceanConfig.implicit_vertical_mixing``.

Backward-Euler vertical mixing is a Lie operator split (1st-order in
time) of the constant + KPP + enhanced-diffusion verticals.  At small
explicit-CFL it must agree with the existing explicit path to a few
parts in ``dt * K / dz²``; at large CFL the explicit path is unstable
and the implicit path must remain finite.

Run with::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_implicit_vertical_mixing.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.vertical_mixing.config import (
    VerticalMixingConfig, ConstantVerticalMixingConfig,
)
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _build_model(implicit: bool, A_v=1e-3, K_v=1e-4,
                 convection=False, n_lat=18, n_lon=36, n_lev=6):
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=n_lev, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_surface=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=4000.0,
    )
    # Add some shear/stratification so vertical diffusion has work to do.
    rng = np.random.RandomState(0)
    T = np.asarray(state.T.data).copy()
    T += rng.normal(0, 0.1, T.shape).astype(T.dtype)
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))

    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="constant",
            constant=ConstantVerticalMixingConfig(A_v=A_v, K_v=K_v),
        ),
        lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
        surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion" if convection else "none",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=1.0, K_bg=1e-5, smooth_transition=False,
            ),
        ),
        shortwave_penetration=None,
    )
    cfg = LatLonCGridOceanConfig(
        # Disable everything that isn't being tested so the comparison
        # is between explicit and implicit vertical mixing only.
        A_h=0.0, B_h=0.0, K_h=0.0, K_bih=0.0,
        bottom_drag_r=0.0, hyperdiff_coeff=0.0,
        # Background floors that the implicit path must still apply.
        A_v=A_v, K_v=K_v,
        n_barotropic_substeps=5,
        physics=physics,
        implicit_vertical_mixing=implicit,
        # Standard barotropic solver path
        barotropic_solver="explicit_substep",
    )
    return LatLonCGridOceanModel(grid, z_coord, cfg), state


class TestExplicitImplicitEquivalence:
    """At small CFL, explicit and implicit must agree closely."""

    def test_tracer_drift_small_cfl(self):
        # Pick dt and K such that CFL = dt*K/dz² is tiny → backward-Euler
        # and forward-Euler differ by O(CFL).  At dt=60s, K=1e-3, surface
        # dz≈200m → CFL ≈ 1.5e-6, well below the float32 noise floor.
        model_e, state_e = _build_model(implicit=False, A_v=1e-3, K_v=1e-3)
        model_i, state_i = _build_model(implicit=True, A_v=1e-3, K_v=1e-3)
        dt = 60.0

        s_e = model_e.step(state_e, dt)
        s_i = model_i.step(state_i, dt)

        T_e = np.asarray(s_e.T.data)
        T_i = np.asarray(s_i.T.data)
        # Compare the two end states directly; at small CFL the implicit
        # and explicit paths must agree to a few × machine epsilon of the
        # tracer values themselves.
        denom_rms = float(np.sqrt(np.mean(T_e ** 2)))
        diff_rms = float(np.sqrt(np.mean((T_i - T_e) ** 2)))
        rel = diff_rms / max(denom_rms, 1.0)
        # Tolerance is set well above float32 noise (~1e-7) but tight
        # enough to catch any structural discrepancy in the operator
        # split (e.g. background K not folded in).
        assert rel < 1e-5, f"implicit vs explicit RMS relative drift {rel}"

    def test_total_T_S_preserved(self):
        """Implicit vertical diffusion has zero-flux BCs → conserves T·dz."""
        model_i, state_i = _build_model(implicit=True, A_v=1e-3, K_v=1e-2)
        dt = 600.0

        s = model_i.step(state_i, dt)
        # The implicit step inside this single ocean step is not the
        # only operator (advection + GM/Redi etc. also run), so check
        # the implicit kernel directly via its public function.
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean, build_dz_half,
        )
        dz = np.full((6,), 4000.0 / 6)
        dzh = build_dz_half(jnp.asarray(dz))
        phi = jnp.linspace(20.0, 2.0, 6)
        K = jnp.full((5,), 1e-1)
        phi_new = implicit_vertical_diffusion_ocean(
            phi, K, jnp.asarray(dz), dzh, dt=3600.0,
        )
        before = float(jnp.sum(phi * jnp.asarray(dz)))
        after = float(jnp.sum(phi_new * jnp.asarray(dz)))
        assert abs(after - before) < 1e-6 * max(abs(before), 1.0)


class TestStabilityAtLargeK:
    """Explicit blows up where CFL>0.5; implicit must stay finite."""

    def test_convective_K_stable(self):
        """With K_conv=1.0 and surface dz≈26m, explicit CFL≈0.44 at dt=300s.

        Implicit must remain finite (and indeed unconditionally stable)
        at the same dt.  We perturb a surface column to be statically
        unstable so convection fires.
        """
        # 20 levels over 5500m → surface dz≈26m (matches production grid)
        grid = create_latlon_grid(n_lat=18, n_lon=36)
        z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0)
        state = rest_state_latlon_cgrid_ocean(
            grid, z_coord, T_surface=20.0, T_deep=2.0,
            S_uniform=35.0, H_max=5500.0,
        )
        # Cold surface, warm below → statically unstable, fires convection.
        T_arr = np.asarray(state.T.data).copy()
        T_arr[..., 0] = -2.0   # surface freezes
        T_arr[..., 1] = 20.0   # warm 2nd layer (instability)
        state = state._replace(T=state.T.replace(data=jnp.asarray(T_arr)))

        physics = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(
                scheme="constant",
                constant=ConstantVerticalMixingConfig(A_v=1e-3, K_v=1e-4),
            ),
            lateral_mixing=type(OceanPhysicsConfig().lateral_mixing)(scheme="none"),
            surface_forcing=type(OceanPhysicsConfig().surface_forcing)(scheme="none"),
            convection=OceanConvectionConfig(
                scheme="enhanced_diffusion",
                enhanced_diffusion=EnhancedDiffusionConfig(
                    K_conv=1.0, K_bg=1e-5, smooth_transition=False,
                ),
            ),
            shortwave_penetration=None,
        )
        cfg = LatLonCGridOceanConfig(
            A_h=0.0, B_h=0.0, K_h=0.0, K_bih=0.0,
            bottom_drag_r=0.0, hyperdiff_coeff=0.0,
            A_v=1e-3, K_v=1e-4,
            n_barotropic_substeps=5,
            physics=physics,
            implicit_vertical_mixing=True,
            barotropic_solver="explicit_substep",
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        dt = 300.0
        s = model.step(state, dt)
        assert bool(jnp.all(jnp.isfinite(s.T.data)))
        # Should partially mix out the instability — surface T should rise
        # toward the second layer's warm value.
        T_sfc_after = float(s.T.data[5, 5, 0])
        assert T_sfc_after > -2.0
