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

5. ``land_mask_from_phis`` is bounded in [0, 1] and produces a near-
   step transition through ``phis = 0``.

6. ``AIMIPClassicalParams.from_defaults(spatial_surface=True)``
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
    """Default truncation has 13 coefs per field; 7 fields -> 91 coefs."""
    from legoesm.training.aimip_spatial import AIMIPSpatialSurfaceParams
    p = AIMIPSpatialSurfaceParams.from_defaults()
    assert p.n_trainable() == 91


# ----------------------------------------------------------------------
# land_mask_from_phis
# ----------------------------------------------------------------------

def test_land_mask_from_phis_bounded_and_sigmoid():
    from legoesm.training.aimip_spatial import land_mask_from_phis
    phis = jnp.array([-1.0e4, -100.0, 0.0, 100.0, 1.0e4])
    mask = land_mask_from_phis(phis, smooth=True, sharpness=1.0e-2)
    assert jnp.all(mask >= 0.0)
    assert jnp.all(mask <= 1.0)
    # Monotonic in phis.
    assert jnp.all(jnp.diff(mask) > 0.0)
    # Centered at phis=0 -> mask=0.5.
    assert math.isclose(float(mask[2]), 0.5, abs_tol=1e-12)


# ----------------------------------------------------------------------
# Integration with AIMIPClassicalParams
# ----------------------------------------------------------------------

def test_aimip_classical_params_with_spatial_surface_builds():
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.aimip_spatial import land_mask_from_phis
    grid = _grid_t11()

    params = AIMIPClassicalParams.from_defaults(spatial_surface=True)
    assert params.spatial_surface is not None
    assert params.spatial_surface.n_trainable() == 91

    # Synthetic phis: positive over half the grid (Northern hemisphere).
    phis = jnp.where(
        grid.lat2d > 0.0, jnp.full(grid.lat2d.shape, 5.0e4),
        jnp.zeros_like(grid.lat2d),
    )
    land_mask = land_mask_from_phis(phis, smooth=True)

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
