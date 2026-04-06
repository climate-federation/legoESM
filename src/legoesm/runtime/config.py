"""Canonical runtime configuration object and bootstrap function.

``RuntimeConfig`` is an immutable snapshot of all runtime decisions:
backend, X64 mode, precision policy, device layout, and distributed
state.  It is created once by :func:`bootstrap` and threaded through
the driver layer so that no module needs to query global singletons
after startup.

Usage
-----
>>> from legoesm.runtime import bootstrap
>>> rc = bootstrap(precision="fp32")
>>> rc.backend        # "cpu"
>>> rc.device_config  # DeviceConfig(...)
>>> rc.precision      # PrecisionPolicy(...)
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class RuntimeConfig(NamedTuple):
    """Immutable snapshot of the resolved runtime environment.

    Attributes
    ----------
    backend : str
        Lowercase backend name (``"cpu"``, ``"gpu"``, ``"tpu"``, ``"metal"``).
    x64 : bool
        Whether JAX x64 mode is active.
    precision : object
        The active :class:`~legoesm.core.precision.PrecisionPolicy`.
    device_config : object
        The :class:`~legoesm.parallel.mesh.DeviceConfig` describing
        the device mesh.
    distributed : bool
        Whether multi-node MPI is in use.
    """
    backend: str
    x64: bool
    precision: object   # PrecisionPolicy (avoiding import for forward-ref)
    device_config: object   # DeviceConfig
    distributed: bool


# Singleton — set once by bootstrap(), queryable afterwards.
_active: RuntimeConfig | None = None


def get_runtime_config() -> RuntimeConfig | None:
    """Return the active ``RuntimeConfig``, or ``None`` if not bootstrapped."""
    return _active


def bootstrap(
    *,
    precision: str = "fp32",
    x64: bool | None = None,
    backend: str | None = None,
    n_devices: int | str = "auto",
    distributed: bool = False,
    grid_type: str = "cubed_sphere",
    configure_xla: bool = True,
) -> RuntimeConfig:
    """One-shot runtime initialisation.

    This must be called **before** any heavy JAX computation.  It:

    1. Applies XLA/backend flags (``configure_backend``).
    2. Enables X64 if the precision mode or *x64* flag requires it.
    3. Activates the precision policy globally.
    4. Creates the device mesh.

    Parameters
    ----------
    precision : str
        ``"fp32"``, ``"fp64"``, or ``"mixed"``.
    x64 : bool or None
        Force X64 on/off.  ``None`` (default) infers from *precision*
        (enabled for ``"fp64"`` and ``"mixed"``).
    backend : str or None
        Force a JAX backend.  ``None`` = auto-detect.
    n_devices : int or ``"auto"``
        Number of devices.
    distributed : bool
        Initialise multi-node MPI.
    grid_type : str
        ``"cubed_sphere"``, ``"latlon"``, or ``"spectral"``.
    configure_xla : bool
        Apply per-backend XLA flags.

    Returns
    -------
    RuntimeConfig
    """
    global _active

    # 1. Backend XLA flags ---------------------------------------------------
    from legoesm.runtime.backend import (
        configure_backend,
        enable_x64,
        is_x64_enabled,
        get_backend,
    )

    if configure_xla:
        resolved_backend = configure_backend(backend)
    else:
        resolved_backend = backend or get_backend()

    # 2. X64 policy ----------------------------------------------------------
    need_x64 = x64
    if need_x64 is None:
        # Infer: fp64, mixed, and mixed_fp64_storage all need x64.
        need_x64 = precision.strip().lower() in (
            "fp64", "float64", "mixed", "mixed_fp64_storage",
        )

    if need_x64:
        enable_x64(quiet=True)

    # 3. Precision policy ----------------------------------------------------
    from legoesm.runtime.precision import apply_precision
    policy = apply_precision(precision)

    # Verify float64 is actually available if required.
    from legoesm.core.precision import validate_policy
    validate_policy(policy)

    # 4. Device mesh ---------------------------------------------------------
    from legoesm.runtime.devices import setup_devices
    device_config = setup_devices(
        n_devices=n_devices,
        backend=backend,
        distributed=distributed,
        grid_type=grid_type,
    )

    # 5. Build immutable snapshot --------------------------------------------
    rc = RuntimeConfig(
        backend=resolved_backend,
        x64=is_x64_enabled(),
        precision=policy,
        device_config=device_config,
        distributed=distributed,
    )
    _active = rc

    _n_dev = getattr(device_config, "n_devices", None)
    if _n_dev is None and isinstance(device_config, dict):
        _n_dev = device_config.get("n_devices", "?")
    logger.info(
        "Runtime bootstrapped: backend=%s, x64=%s, precision=%s, "
        "devices=%s, distributed=%s",
        rc.backend,
        rc.x64,
        precision,
        _n_dev,
        rc.distributed,
    )
    return rc


def bootstrap_from_yaml_config(config) -> RuntimeConfig:
    """Bootstrap from a legoESM YAML ``Config`` object.

    Reads ``hardware.precision.*``, ``hardware.parallelism.*``, and
    ``hardware.devices`` keys and delegates to :func:`bootstrap`.

    This replaces the old ``core.hardware.apply_hardware_config``.
    """
    # Precision mode: prefer explicit mode key, fall back to legacy 3-component.
    explicit_mode = config.get("hardware.precision.mode", None)
    if explicit_mode is not None:
        precision = str(explicit_mode).strip().lower()
    else:
        dynamics_prec = config.get("hardware.precision.dynamics", None)
        if dynamics_prec is not None:
            prec_str = str(dynamics_prec).strip().lower()
            if prec_str in ("float64", "fp64", "double"):
                precision = "fp64"
            else:
                # Check conservation for mixed hint.
                cons = config.get("hardware.precision.conservation", None)
                if cons is not None and str(cons).strip().lower() in ("float64", "fp64"):
                    precision = "mixed"
                else:
                    precision = "fp32"
        else:
            precision = "fp32"

    n_devices = config.get("hardware.parallelism.n_devices", "auto")
    backend = config.get("hardware.parallelism.backend", None)
    distributed = bool(config.get("hardware.parallelism.distributed", False))

    # Legacy compat: hardware.devices as n_devices fallback.
    legacy_devices = config.get("hardware.devices", "auto")
    if n_devices in (None, "auto") and legacy_devices not in (None, "auto"):
        n_devices = legacy_devices
    if n_devices is None:
        n_devices = "auto"

    grid_type = config.get("grid.type", "cubed_sphere")

    return bootstrap(
        precision=precision,
        backend=backend,
        n_devices=n_devices,
        distributed=distributed,
        grid_type=grid_type,
    )
