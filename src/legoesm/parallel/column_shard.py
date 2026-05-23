"""Horizontal column sharding for per-column physics.

Issue #273 follow-up to the level-parallel cubed-sphere fallback in
``legoesm.parallel.mesh``: level sharding is the *wrong* axis for the
column-wise radiation, convection, and turbulence schemes that drive
the AMIP throughput gap.  Each column is independent of every other
column, so flattening the horizontal axis (face × i × j → flat) and
sharding *that* axis across devices yields embarrassingly-parallel
execution at arbitrary device counts — no face-divisibility
constraint, no inter-device halo exchange for the radiation step.

The helpers in this module are deliberately small:

* ``create_column_mesh(n_devices)`` returns a ``Mesh`` with a single
  ``'col'`` axis.  Works for any positive ``n_devices``.
* ``shard_columns(field, mesh)`` places a column-major array (leading
  axis = column index) with the column axis sharded.
* ``replicate(field, mesh)`` places a scalar-or-config field
  replicated across the mesh.
* ``pad_to_shardable(ncol, n_devices)`` returns ``(padded_ncol,
  pad_amount)`` — JAX sharding requires the axis size to be divisible
  by the device count along that axis, so callers pad the column
  arrays before sharding and slice back to ``ncol`` after gathering.

The dycore stays on whatever cubed-sphere face mesh it was already
using.  Only the per-column physics step is wrapped — radiation
becomes ``shard_columns → vmap-over-cols (already in the column
function) → unshard``, with the heavy compute distributed across
all devices on the node regardless of face-sharding divisibility.

Status: Phase 3 follow-up scaffolding for issue #273.  Operates
independently of the cubed-sphere face mesh; intended caller is
the radiation integration layer (see
``atmosphere.physics.radiation.integration``).  Full driver
integration — radiation step wrapped to call ``shard_columns``
before dispatch and to gather after — is the next step.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


__all__ = (
    "create_column_mesh",
    "shard_columns",
    "replicate",
    "pad_to_shardable",
)


def create_column_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> Mesh:
    """Create a single-axis ``Mesh`` for horizontal-column sharding.

    Parameters
    ----------
    n_devices : int or ``"auto"``
        Number of devices.  ``"auto"`` uses all available.
    backend : str or None
        JAX backend.  ``None`` = auto.
    devices : sequence or None
        Optional explicit device list.

    Returns
    -------
    jax.sharding.Mesh
        Single axis ``'col'``.  Suitable for any positive
        ``n_devices`` — there is no divisibility constraint on the
        device count itself; the *array* axis size must be padded
        to a multiple of ``n_devices`` before sharding (see
        ``pad_to_shardable``).
    """
    if devices is not None:
        devices = list(devices)
    elif backend is not None:
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            devices = jax.devices()
    else:
        devices = jax.devices()

    all_count = len(devices)
    if n_devices == "auto":
        n_dev = all_count
    else:
        n_dev = min(int(n_devices), all_count)

    if n_dev < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_dev}")
    selected = devices[:n_dev]
    return Mesh(selected, axis_names=("col",))


def pad_to_shardable(ncol: int, n_devices: int) -> tuple[int, int]:
    """Return ``(padded_ncol, pad_amount)`` such that ``padded_ncol``
    is the smallest multiple of ``n_devices`` that is ``>= ncol``.

    JAX sharding requires axis size divisibility; padding by zero
    values + slicing the result back to ``ncol`` lets a sharded
    per-column physics step run on arbitrary ``(ncol, n_devices)``
    combinations.  The padded entries carry zero physical signal
    and are discarded on gather.
    """
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if ncol < 0:
        raise ValueError(f"ncol must be >= 0, got {ncol}")
    rem = ncol % n_devices
    if rem == 0:
        return ncol, 0
    pad = n_devices - rem
    return ncol + pad, pad


def shard_columns(field: jax.Array, mesh: Mesh) -> jax.Array:
    """Place ``field`` with leading axis sharded across ``mesh['col']``.

    The leading axis is treated as the column index.  Trailing axes
    (vertical levels, bands, etc.) are replicated on each device.

    The caller is responsible for ensuring ``field.shape[0]`` divides
    ``mesh.shape['col']`` — use ``pad_to_shardable`` first if not.
    """
    if field.ndim == 0:
        return jax.device_put(field, NamedSharding(mesh, P()))
    trailing = (None,) * (field.ndim - 1)
    return jax.device_put(field, NamedSharding(mesh, P("col", *trailing)))


def replicate(field: jax.Array, mesh: Mesh) -> jax.Array:
    """Place ``field`` replicated across every device in ``mesh``."""
    return jax.device_put(field, NamedSharding(mesh, P()))
