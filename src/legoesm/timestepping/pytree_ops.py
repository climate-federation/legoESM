"""Shared pytree arithmetic utilities for time integrators."""

from __future__ import annotations

import jax


def pytree_axpy(x, y, alpha):
    """Compute x + alpha * y for two pytrees with the same structure."""
    return jax.tree.map(lambda xi, yi: xi + alpha * yi, x, y)


def pytree_linear_combination(x, y, a, b):
    """Compute a * x + b * y for two pytrees with the same structure."""
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)
