"""Device detection and mesh setup — thin delegation layer.

The actual mesh creation logic lives in ``parallel.mesh`` and
``parallel.device_config``.  This module provides a unified API that
:func:`runtime.bootstrap` can call without the caller needing to know
which sub-module to import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Re-export HardwareConfig from parallel.device_config for convenience.
from legoesm.parallel.device_config import (        # noqa: F401
    HardwareConfig,
    detect_devices as detect_hardware,
    MixedPrecisionPolicy,
    mixed_precision_policy,
    get_optimal_dtype,
)

# Re-export DeviceConfig and mesh helpers from parallel.mesh.
from legoesm.parallel.mesh import (                  # noqa: F401
    DeviceConfig,
    get_active_config,
    set_active_config,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    shard_pytree,
    replicate_pytree,
)


def setup_devices(
    *,
    n_devices: int | str = "auto",
    backend: str | None = None,
    distributed: bool = False,
    grid_type: str = "cubed_sphere",
) -> DeviceConfig:
    """One-shot device setup.

    Selects the appropriate mesh constructor based on *grid_type* and
    *distributed* flag.

    Returns a :class:`DeviceConfig` (which is also set as the active
    singleton via ``parallel.mesh.set_active_config``).
    """
    # Late imports so that ``unittest.mock.patch`` on the canonical
    # ``legoesm.parallel.mesh`` module works in tests.
    from legoesm.parallel import mesh as _mesh

    if distributed:
        from legoesm.parallel.distributed import initialize_distributed
        return initialize_distributed(grid_type=grid_type)

    if grid_type == "latlon":
        return _mesh.create_latlon_mesh(n_devices=n_devices, backend=backend)
    elif grid_type == "spectral":
        return _mesh.create_level_mesh(n_devices=n_devices, backend=backend)
    else:
        return _mesh.create_device_mesh(n_devices=n_devices, backend=backend)
