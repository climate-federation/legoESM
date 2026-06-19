"""JAX-version compatibility shim for ``shard_map``.

``shard_map`` graduated from ``jax.experimental.shard_map`` to the top-level
``jax`` namespace in JAX >= 0.8.  Re-exporting it here gives every parallel
backend a single import site that works across both layouts, instead of
copy-pasting the ``try/except`` block in each module.

Usage::

    from legoesm.parallel.shard_map_compat import shard_map
"""
from __future__ import annotations

try:  # JAX >= 0.8 top-level export
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX fallback
    from jax.experimental.shard_map import shard_map

__all__ = ["shard_map"]
