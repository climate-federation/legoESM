"""Direct unit test for the shared zero-flux divergence kernel (#518 item 4).

`flux_divergence_zero_flux` is the single divergence core now shared by the four
vertical zero-flux diffusion stencils (vertical_diffusion, vertical_diffusion_
variable_K, mpas_integration._vertical_diffusion_edge_partial, and
_gm_redi_common.vertical_flux_divergence).  This pins it bit-identical to the
original `concatenate([top, interior, bottom])` triad and to the GM/Redi
`max(dz, eps)`-floored pad form for arbitrary (non-zero-flux) inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.mixing import (
    flux_divergence_zero_flux,
    vertical_diffusion,
    vertical_diffusion_variable_K,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    vertical_flux_divergence,
)

jax.config.update("jax_enable_x64", True)


class _ZCoord:
    """Minimal stand-in: the vertical-diffusion stencils read only ``dz_ref``."""

    def __init__(self, dz_ref):
        self.dz_ref = dz_ref


def _triad_reference(flux, dz_safe):
    """Exact pre-#518 concatenate([top, interior, bottom]) construction."""
    top = -flux[..., :1] / dz_safe[..., :1]
    interior = (flux[..., :-1] - flux[..., 1:]) / dz_safe[..., 1:-1]
    bottom = flux[..., -1:] / dz_safe[..., -1:]
    return jnp.concatenate([top, interior, bottom], axis=-1)


def _gm_pad_reference(F_z, dz_actual, eps):
    """Exact pre-#518 GM/Redi pad form."""
    pad_axes = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)


def test_matches_triad_construction():
    rng = np.random.default_rng(0)
    flux = jnp.asarray(rng.standard_normal((5, 8, 7)))   # (..., nlev-1)
    dz_safe = jnp.asarray(1.0 + np.abs(rng.standard_normal((5, 8, 8))))
    got = flux_divergence_zero_flux(flux, dz_safe)
    ref = _triad_reference(flux, dz_safe)
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_matches_gm_redi_pad_form():
    rng = np.random.default_rng(1)
    F_z = jnp.asarray(rng.standard_normal((4, 9)))
    dz_actual = jnp.asarray(rng.standard_normal((4, 10)) * 2.0)  # incl <=0
    eps = 1e-12
    got = vertical_flux_divergence(F_z, dz_actual, eps)
    ref = _gm_pad_reference(F_z, dz_actual, eps)
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def _zcoord(nlev):
    return _ZCoord(jnp.asarray(np.linspace(10.0, 100.0, nlev)))


def test_full_operator_scalar_unchanged():
    """vertical_diffusion output is bit-identical to its old triad form."""
    rng = np.random.default_rng(2)
    nlev = 6
    field = jnp.asarray(rng.standard_normal((3, 4, nlev)))
    jacobian = jnp.asarray(1.0 + 0.2 * rng.random((3, 4)))
    z_coord = _zcoord(nlev)
    coeff = 1e-3

    out = vertical_diffusion(field, z_coord, jacobian, coeff)

    # Reconstruct via the old triad path.
    dz = z_coord.dz_ref * jacobian[..., None]
    dz_half = 0.5 * (dz[..., :-1] + dz[..., 1:])
    dz_half_safe = jnp.where(dz_half > 0.0, dz_half, 1.0)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    field_safe = jnp.where(dz > 0.0, field, 0.0)
    df_dz = (field_safe[..., :-1] - field_safe[..., 1:]) / dz_half_safe
    flux = jnp.asarray(coeff, field.dtype) * df_dz
    ref = jnp.where(dz > 0.0, _triad_reference(flux, dz_safe), 0.0)
    assert np.array_equal(np.asarray(out), np.asarray(ref))


def test_full_operator_array_K_unchanged():
    rng = np.random.default_rng(3)
    nlev = 6
    field = jnp.asarray(rng.standard_normal((2, 5, nlev)))
    jacobian = jnp.asarray(1.0 + 0.2 * rng.random((2, 5)))
    K_half = jnp.asarray(1e-3 * np.abs(rng.standard_normal((2, 5, nlev - 1))))
    z_coord = _zcoord(nlev)

    out = vertical_diffusion_variable_K(field, z_coord, jacobian, K_half)

    dz = z_coord.dz_ref * jacobian[..., None]
    dz_half = 0.5 * (dz[..., :-1] + dz[..., 1:])
    dry_iface = (dz[..., :-1] <= 0.0) | (dz[..., 1:] <= 0.0)
    K = jnp.where(dry_iface, 0.0, K_half)
    dz_half_safe = jnp.where(dz_half > 0.0, dz_half, 1.0)
    dz_safe = jnp.where(dz > 0.0, dz, 1.0)
    field_safe = jnp.where(dz > 0.0, field, 0.0)
    df_dz = (field_safe[..., :-1] - field_safe[..., 1:]) / dz_half_safe
    flux = K * df_dz
    ref = jnp.where(dz > 0.0, _triad_reference(flux, dz_safe), 0.0)
    assert np.array_equal(np.asarray(out), np.asarray(ref))


def test_surface_sign_of_zero_preserved():
    """The shared kernel uses -flux[:1] (not a padded 0-flux[0]) so the surface
    term keeps -0.0 when flux[0]==+0.0, byte-identical to the old triad."""
    flux = jnp.zeros((1, 3))            # flux[...,0] == +0.0
    dz = jnp.ones((1, 4))
    out = np.asarray(flux_divergence_zero_flux(flux, dz))
    ref = np.asarray(_triad_reference(flux, dz))
    # signbit must match element-wise (catches +0.0 vs -0.0).
    assert np.array_equal(np.signbit(out), np.signbit(ref))
    assert np.signbit(out[0, 0])  # surface term is -0.0


def test_differentiable():
    rng = np.random.default_rng(4)
    flux = jnp.asarray(rng.standard_normal((3, 5)))
    dz = jnp.asarray(1.0 + np.abs(rng.standard_normal((3, 6))))
    g = jax.grad(lambda f: jnp.sum(flux_divergence_zero_flux(f, dz) ** 2))(flux)
    assert np.all(np.isfinite(np.asarray(g)))
