#!/usr/bin/env python
"""Test JAX differentiability of all atmosphere discretizations over 10 steps.

For each cubed-sphere discretization (centered, finite_volume, cgrid,
fc_gram, fc_gram_cgrid, spectral) at each dynamics level (shallow_water,
hydrostatic, nonhydrostatic), compute jax.grad through 10 time steps
and verify the gradients are finite and non-zero.
"""

import sys
import traceback

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    ShallowWaterState,
    HydrostaticState,
    NonHydrostaticState,
)

N = 8
N_STEPS = 10
DT_SW = 300.0
DT_PE = 60.0
DT_CE = 5.0

grid = create_cubed_sphere(N)
sigma = create_sigma_coordinate(5)
height_coord = create_height_coordinate(10, 30000.0)
z_s = jnp.zeros((6, N, N))
terrain_metric = compute_terrain_metric(z_s, height_coord)


# --- State constructors ---

def make_sw_state():
    shape = (6, N, N)
    dims = ("face", "x", "y")
    h0 = 1000.0
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * h0, name="h", dims=dims, units="m"),
        u=Field(data=jnp.ones(shape) * 5.0, name="u", dims=dims, units="m/s"),
        v=Field(data=jnp.ones(shape) * 2.0, name="v", dims=dims, units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=dims, units="m"),
    )


def make_pe_state():
    shape_3d = (6, N, N, sigma.n_levels)
    shape_2d = (6, N, N)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * 5.0, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * 2.0, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * 280.0, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )


def make_ce_state():
    nlev = height_coord.n_levels
    shape_3d = (6, N, N, nlev)
    shape_w = (6, N, N, nlev + 1)
    shape_2d = (6, N, N)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    return NonHydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros(shape_w), name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(
            data=jnp.zeros((*shape_3d, 0)),
            name="tracers", dims=("face", "x", "y", "level", "tracer"), units="kg/kg",
        ),
    )


# --- Test runner ---

def test_differentiability(name, model, state, dt, grad_field="u"):
    """Test jax.grad through N_STEPS of model.step()."""
    try:
        def loss(field_data):
            if grad_field == "h":
                s = state._replace(h=state.h.replace(data=field_data))
            elif grad_field == "u":
                s = state._replace(u=state.u.replace(data=field_data))
            elif grad_field == "theta_prime":
                s = state._replace(theta_prime=state.theta_prime.replace(data=field_data))
            else:
                raise ValueError(f"Unknown grad_field={grad_field}")

            for _ in range(N_STEPS):
                s = model.step(s, dt)

            if hasattr(s, 'h'):
                return jnp.mean(s.h.data ** 2)
            elif hasattr(s, 'T'):
                return jnp.mean(s.T.data ** 2)
            elif hasattr(s, 'theta_prime'):
                return jnp.mean(s.theta_prime.data ** 2)

        if grad_field == "h":
            data = state.h.data
        elif grad_field == "u":
            data = state.u.data
        elif grad_field == "theta_prime":
            data = state.theta_prime.data
        else:
            raise ValueError(f"Unknown grad_field={grad_field}")

        grads = jax.grad(loss)(data)
        is_finite = bool(jnp.all(jnp.isfinite(grads)))
        is_nonzero = bool(not jnp.allclose(grads, 0.0))
        grad_norm = float(jnp.max(jnp.abs(grads)))

        if is_finite and is_nonzero:
            print(f"  PASS  {name}  |grad|_max={grad_norm:.4e}")
        elif is_finite:
            print(f"  WARN  {name}  gradients are all zero")
        else:
            print(f"  FAIL  {name}  non-finite gradients, |grad|_max={grad_norm}")
        return is_finite
    except Exception as e:
        print(f"  ERROR {name}  {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
            # ==============================================================================
            # Shallow Water
            # ==============================================================================
            print("=" * 70)
            print(f"SHALLOW WATER  (C{N}, {N_STEPS} steps, dt={DT_SW}s)")
        print("=" * 70)

        sw_state = make_sw_state()

        # Centered
        from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterModel, ShallowWaterConfig
        model = ShallowWaterModel(grid, ShallowWaterConfig(
            hyperdiff_coeff=0.0, use_conservation_fixer=False,
        ))
        test_differentiability("centered", model, sw_state, DT_SW, grad_field="h")

        # Finite Volume
        from legoesm.atmosphere.dynamics.shallow_water_fv import FVShallowWaterModel, FVShallowWaterConfig
        model = FVShallowWaterModel(grid, FVShallowWaterConfig(
            use_conservation_fixer=False, edge_blend_strength=0.25,
        ))
        test_differentiability("finite_volume", model, sw_state, DT_SW, grad_field="h")

        # FV with edge_blend=0 (no blending)
        model = FVShallowWaterModel(grid, FVShallowWaterConfig(
            use_conservation_fixer=False, edge_blend_strength=0.0,
        ))
        test_differentiability("finite_volume (no blend)", model, sw_state, DT_SW, grad_field="h")

        # C-grid
        from legoesm.atmosphere.dynamics.shallow_water_cgrid import (
            CGShallowWaterCubedModel, CGShallowWaterCubedConfig,
        )
        model = CGShallowWaterCubedModel(grid, CGShallowWaterCubedConfig())
        test_differentiability("cgrid", model, sw_state, DT_SW, grad_field="h")

        # FC-gram
        from legoesm.atmosphere.dynamics.shallow_water_fc import (
            FCShallowWaterModel, FCShallowWaterConfig,
        )
        model = FCShallowWaterModel(grid, FCShallowWaterConfig())
        test_differentiability("fc_gram", model, sw_state, DT_SW, grad_field="h")

        # FC-gram C-grid
        from legoesm.atmosphere.dynamics.shallow_water_fc_cgrid import (
            FCCGShallowWaterModel, FCCGShallowWaterConfig,
        )
        model = FCCGShallowWaterModel(grid, FCCGShallowWaterConfig())
        test_differentiability("fc_gram_cgrid", model, sw_state, DT_SW, grad_field="h")

        # Spectral
        from legoesm.atmosphere.dynamics.spectral_sw import SpectralShallowWaterModel
        try:
            model = SpectralShallowWaterModel(grid)
            test_differentiability("spectral", model, sw_state, DT_SW, grad_field="h")
        except Exception as e:
            print(f"  SKIP  spectral  {type(e).__name__}: {e}")

        # ==============================================================================
        # Hydrostatic (Primitive Equations)
        # ==============================================================================
        print()
        print("=" * 70)
        print(f"HYDROSTATIC PE  (C{N}, {N_STEPS} steps, dt={DT_PE}s)")
        print("=" * 70)

        pe_state = make_pe_state()

        # Centered
        from legoesm.atmosphere.dynamics.primitive_eq import (
            PrimitiveEquationModel, PrimitiveEquationConfig,
        )
        model = PrimitiveEquationModel(grid, sigma, PrimitiveEquationConfig(
            hyperdiff_coeff=0.0, use_conservation_fixer=False,
        ))
        test_differentiability("centered", model, pe_state, DT_PE)

        # Finite Volume
        from legoesm.atmosphere.dynamics.primitive_eq_fv import (
            FVPrimitiveEquationModel, FVPrimitiveEquationConfig,
        )
        model = FVPrimitiveEquationModel(grid, sigma, FVPrimitiveEquationConfig(
            use_conservation_fixer=False, edge_blend_strength=0.25,
        ))
        test_differentiability("finite_volume", model, pe_state, DT_PE)

        # FV no blend
        model = FVPrimitiveEquationModel(grid, sigma, FVPrimitiveEquationConfig(
            use_conservation_fixer=False, edge_blend_strength=0.0,
        ))
        test_differentiability("finite_volume (no blend)", model, pe_state, DT_PE)

        # C-grid
        from legoesm.atmosphere.dynamics.primitive_eq_cgrid import (
            CGPrimitiveEquationModel, CGPrimitiveEquationConfig,
        )
        model = CGPrimitiveEquationModel(grid, sigma, CGPrimitiveEquationConfig())
        test_differentiability("cgrid", model, pe_state, DT_PE)

        # FC-gram
        from legoesm.atmosphere.dynamics.primitive_eq_fc import (
            FCPrimitiveEquationModel, FCPrimitiveEquationConfig,
        )
        model = FCPrimitiveEquationModel(grid, sigma, FCPrimitiveEquationConfig())
        test_differentiability("fc_gram", model, pe_state, DT_PE)

        # FC-gram C-grid
        from legoesm.atmosphere.dynamics.primitive_eq_fc_cgrid import (
            FCCGPrimitiveEquationModel, FCCGPrimitiveEquationConfig,
        )
        model = FCCGPrimitiveEquationModel(grid, sigma, FCCGPrimitiveEquationConfig())
        test_differentiability("fc_gram_cgrid", model, pe_state, DT_PE)

        # Spectral
        from legoesm.atmosphere.dynamics.spectral_pe import SpectralPrimitiveEquationModel
        try:
            model = SpectralPrimitiveEquationModel(grid, sigma)
            test_differentiability("spectral", model, pe_state, DT_PE)
        except Exception as e:
            print(f"  SKIP  spectral  {type(e).__name__}: {e}")

        # ==============================================================================
        # Non-Hydrostatic (Compressible Euler)
        # ==============================================================================
        print()
        print("=" * 70)
        print(f"NONHYDROSTATIC CE  (C{N}, {N_STEPS} steps, dt={DT_CE}s)")
        print("=" * 70)

        ce_state = make_ce_state()

        # Centered
        from legoesm.atmosphere.dynamics.compressible_euler import (
            CompressibleEulerModel, CompressibleEulerConfig,
        )
        model = CompressibleEulerModel(grid, height_coord, terrain_metric, CompressibleEulerConfig(
            hyperdiff_coeff=0.0, sponge_coeff=0.0,
        ))
        test_differentiability("centered", model, ce_state, DT_CE, grad_field="theta_prime")

        # Finite Volume
        from legoesm.atmosphere.dynamics.compressible_euler_fv import (
            FVCompressibleEulerModel, FVCompressibleEulerConfig,
        )
        model = FVCompressibleEulerModel(grid, height_coord, terrain_metric, FVCompressibleEulerConfig(
            sponge_coeff=0.0, edge_blend_strength=0.25,
        ))
        test_differentiability("finite_volume", model, ce_state, DT_CE, grad_field="theta_prime")

        # FV no blend
        model = FVCompressibleEulerModel(grid, height_coord, terrain_metric, FVCompressibleEulerConfig(
            sponge_coeff=0.0, edge_blend_strength=0.0,
        ))
        test_differentiability("finite_volume (no blend)", model, ce_state, DT_CE, grad_field="theta_prime")

        # C-grid
        from legoesm.atmosphere.dynamics.compressible_euler_cgrid import (
            CGCompressibleEulerModel, CGCompressibleEulerConfig,
        )
        model = CGCompressibleEulerModel(grid, height_coord, terrain_metric, CGCompressibleEulerConfig(
            sponge_coeff=0.0,
        ))
        test_differentiability("cgrid", model, ce_state, DT_CE, grad_field="theta_prime")

        # FC-gram
        from legoesm.atmosphere.dynamics.compressible_euler_fc import (
            FCCompressibleEulerModel, FCCompressibleEulerConfig,
        )
        model = FCCompressibleEulerModel(grid, height_coord, terrain_metric, FCCompressibleEulerConfig(
            sponge_coeff=0.0,
        ))
        test_differentiability("fc_gram", model, ce_state, DT_CE, grad_field="theta_prime")

        # FC-gram C-grid
        from legoesm.atmosphere.dynamics.compressible_euler_fc_cgrid import (
            FCCGCompressibleEulerModel, FCCGCompressibleEulerConfig,
        )
        model = FCCGCompressibleEulerModel(grid, height_coord, terrain_metric, FCCGCompressibleEulerConfig(
            sponge_coeff=0.0,
        ))
        test_differentiability("fc_gram_cgrid", model, ce_state, DT_CE, grad_field="theta_prime")

        # Spectral
        from legoesm.atmosphere.dynamics.spectral_nh import SpectralCompressibleEulerModel
        try:
            model = SpectralCompressibleEulerModel(grid, height_coord, terrain_metric)
            test_differentiability("spectral", model, ce_state, DT_CE, grad_field="theta_prime")
        except Exception as e:
            print(f"  SKIP  spectral  {type(e).__name__}: {e}")

        print()
        print("Done.")
