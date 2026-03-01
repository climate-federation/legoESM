"""Device mesh creation and array sharding for cubed-sphere parallelism.

The cubed sphere has 6 faces — the natural parallelism boundary.
This module creates a JAX device mesh that shards the face dimension
across available devices, enabling transparent multi-GPU / multi-CPU
execution without changes to operator code.

For single-node multi-device, JAX's XLA compiler automatically inserts
inter-device transfers when a JIT-compiled function accesses data from
another shard. No changes to the halo exchange are needed.

For multi-node MPI, see :mod:`legoesm.parallel.distributed`.
"""

from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

# Number of cubed-sphere faces.
_N_FACES = 6


# ==============================================================================
# Device configuration
# ==============================================================================

class DeviceConfig(NamedTuple):
    """Resolved device configuration.

    Attributes
    ----------
    mesh : Mesh or None
        JAX device mesh.  ``None`` for single-device execution.
    face_sharding : NamedSharding or None
        Sharding specification for face-first arrays ``(6, n, n, ...)``.
        ``None`` for single-device execution.
    replicated_sharding : NamedSharding or None
        Sharding that replicates data across all devices.
        ``None`` for single-device execution.
    n_devices : int
        Number of devices in use.
    backend : str
        Backend name (``"CPU"``, ``"GPU"``, ``"TPU"``, ``"METAL"``).
    is_distributed : bool
        ``True`` if using multi-node MPI.
    """
    mesh: Mesh | None
    face_sharding: NamedSharding | None
    replicated_sharding: NamedSharding | None
    n_devices: int
    backend: str
    is_distributed: bool


# Singleton — set once at startup, queried by the rest of the code.
_active_config: DeviceConfig | None = None


def get_active_config() -> DeviceConfig | None:
    """Return the active device configuration, or ``None``."""
    return _active_config


# ==============================================================================
# Mesh creation
# ==============================================================================

def create_device_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
) -> DeviceConfig:
    """Create a JAX device mesh for cubed-sphere parallelism.

    Parameters
    ----------
    n_devices : int or ``"auto"``
        Number of devices to use.  ``"auto"`` uses all available.
        Must divide 6 (i.e., 1, 2, 3, or 6).  If more than 6 devices
        are available, 6 are used and the remainder is ignored.
    backend : str or None
        JAX backend to use (``"cpu"``, ``"gpu"``, ``"tpu"``).
        ``None`` auto-detects.

    Returns
    -------
    DeviceConfig
        Resolved configuration with mesh, shardings, and metadata.
    """
    global _active_config

    # Resolve devices.
    if backend is not None:
        devices = jax.devices(backend)
    else:
        devices = jax.devices()

    all_count = len(devices)
    backend_name = jax.default_backend().upper()

    # Guard: no devices available.
    if all_count == 0:
        raise RuntimeError(
            f"No JAX devices found"
            + (f" for backend '{backend}'" if backend is not None else "")
            + ". Check your JAX installation and hardware."
        )

    # Resolve n_devices.
    if n_devices == "auto":
        n_dev = min(all_count, _N_FACES)
    else:
        n_dev = int(n_devices)

    # Validate.
    if n_dev > _N_FACES:
        warnings.warn(
            f"Requested {n_dev} devices but only {_N_FACES} cubed-sphere "
            f"faces exist.  Using {_N_FACES} devices.  Sub-face sharding "
            f"is not yet supported.",
            RuntimeWarning,
            stacklevel=2,
        )
        n_dev = _N_FACES

    if _N_FACES % n_dev != 0:
        raise ValueError(
            f"n_devices={n_dev} does not evenly divide {_N_FACES} faces. "
            f"Choose 1, 2, 3, or 6."
        )

    # Single-device fast path — no mesh overhead.
    if n_dev == 1:
        config = DeviceConfig(
            mesh=None,
            face_sharding=None,
            replicated_sharding=None,
            n_devices=1,
            backend=backend_name,
            is_distributed=False,
        )
        _active_config = config
        logger.info("legoESM: single-device mode (%s)", backend_name)
        return config

    # Multi-device: create mesh.
    selected = devices[:n_dev]
    mesh = Mesh(selected, axis_names=("face",))
    face_sharding = NamedSharding(mesh, P("face"))
    replicated_sharding = NamedSharding(mesh, P())

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=face_sharding,
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device mesh on %s (faces per device: %d)",
        n_dev, backend_name, _N_FACES // n_dev,
    )
    return config


# ==============================================================================
# Pytree sharding utilities
# ==============================================================================

def shard_pytree(pytree, config: DeviceConfig):
    """Shard a pytree across devices by the face dimension.

    Arrays whose first axis has size 6 are sharded across the face
    mesh axis.  All other arrays (scalars, 1-D, or non-face-leading)
    are replicated.

    Parameters
    ----------
    pytree
        Any JAX pytree (e.g., a ``ShallowWaterState``).
    config : DeviceConfig
        Output of :func:`create_device_mesh`.

    Returns
    -------
    Sharded pytree with the same structure.
    """
    if config.face_sharding is None:
        return pytree  # single-device: nothing to do

    def _shard_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        if leaf.ndim >= 1 and leaf.shape[0] == _N_FACES:
            return jax.device_put(leaf, config.face_sharding)
        # Replicate non-face arrays (scalars, 1-D, etc.)
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_shard_leaf, pytree)


def replicate_pytree(pytree, config: DeviceConfig):
    """Replicate a pytree across all devices.

    Use this for data that every device needs (e.g., grid metrics).

    Parameters
    ----------
    pytree
        Any JAX pytree.
    config : DeviceConfig
        Output of :func:`create_device_mesh`.

    Returns
    -------
    Replicated pytree.
    """
    if config.replicated_sharding is None:
        return pytree

    def _replicate_leaf(leaf):
        if not isinstance(leaf, jnp.ndarray):
            return leaf
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_replicate_leaf, pytree)
