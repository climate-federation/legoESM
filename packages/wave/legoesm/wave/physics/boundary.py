"""Spectral boundary handling for the wave component.

Port of the absorbing-boundary path of WADEN 46aadc7
``src/waden/boundary.py`` (originally NeuralSWM ``physics_jax/boundary.py``).
The source also exports periodic and
replicate variants; the released configuration uses absorbing edges
only, so the port exposes exactly that.  A run that needs open lateral
boundaries supplies ``boundary_condition`` (zero is the absorbing case).

The source built its boundary condition from the target spectrum of the
validation data (``utils/distributed_padding.py``).  That is label
leakage and is deliberately not ported: the coupled component has no
target spectrum, so the caller passes either zeros or an external
boundary product explicitly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Zero absorbing ghost cells at the domain edge and mask land columns. "
        "An optional external boundary spectrum overwrites the edge cells."
    ),
    "inputs": {
        "spectra": "(B,Nlon,Nlat,Ndir,Nfreq)",
        "boundary_condition": "edge spectra, same shape; zeros = absorbing",
        "mask": "True where sea, (B,Nlon,Nlat)",
    },
    "outputs": {"spectra": "filtered spectrum, same shape"},
    "sign_convention": "Not applicable: no flux is computed here.",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/boundary.py (absorbing path).",
    "idealized_test": (
        "tests/wave/unit/test_boundary.py::test_prescribed_zero_boundary_is_enforced"),
}


def update_absorb_boundary(array: jax.Array) -> jax.Array:
    """Keep the interior and zero every spatial edge cell."""
    interior = jnp.zeros_like(array, dtype=bool)
    interior = interior.at[:, 1:-1, 1:-1, ...].set(True)
    return jnp.where(interior, array, jnp.zeros((), dtype=array.dtype))


def apply_boundary_condition(
    spectra: jax.Array,
    boundary_condition: jax.Array,
    mask: jax.Array,
    boundary_mask: jax.Array | None = None,
) -> jax.Array:
    """Mask land cells and overwrite non-zero boundary cells."""
    mask = jnp.asarray(mask, dtype=bool)
    while mask.ndim < 5:
        mask = mask[..., None]
    mask_expanded = jnp.broadcast_to(mask, spectra.shape)
    if boundary_mask is None:
        boundary_mask = jnp.sum(boundary_condition, axis=(-1, -2)) > 0
    boundary_mask = jnp.asarray(boundary_mask, dtype=bool)
    while boundary_mask.ndim < 5:
        boundary_mask = boundary_mask[..., None]
    boundary_mask_expanded = jnp.broadcast_to(boundary_mask, spectra.shape)
    spectra_masked = jnp.where(
        mask_expanded, spectra, jnp.zeros((), dtype=spectra.dtype))
    return jnp.where(
        boundary_mask_expanded, boundary_condition, spectra_masked)


def identity_boundary_update(array: jax.Array) -> jax.Array:
    """WADEN all-physical-cell layout: retain the physical edge cells."""
    return array


__all__ = ["identity_boundary_update", "apply_boundary_condition", "update_absorb_boundary"]
