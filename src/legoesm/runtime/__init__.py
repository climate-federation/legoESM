"""Canonical runtime bootstrap for legoESM.

This package is the **single source of truth** for:

* Backend detection and XLA configuration
* X64 / precision policy
* Device mesh / layout resolution
* Distributed (MPI) initialisation

Usage
-----
Call :func:`bootstrap` once at the top of every entry point,
**before** importing JAX-heavy model code::

    from legoesm.runtime import bootstrap

    rc = bootstrap(precision="fp32")   # or "fp64", "mixed"
    # rc.backend, rc.precision, rc.device_config now available

For YAML-driven runs::

    from legoesm.runtime import bootstrap_from_yaml_config
    rc = bootstrap_from_yaml_config(config)
"""

from legoesm.runtime.config import (       # noqa: F401
    RuntimeConfig,
    bootstrap,
    bootstrap_from_yaml_config,
    get_runtime_config,
)

from legoesm.runtime.backend import (      # noqa: F401
    get_backend,
    supports_float64,
    enable_x64,
    is_x64_enabled,
    require_x64,
    check_spectral_backend,
    configure_backend,
)

from legoesm.runtime.precision import (    # noqa: F401
    PrecisionPolicy,
    resolve_precision,
    apply_precision,
    get_policy,
    set_policy,
    cast,
    const,
)

from legoesm.runtime.devices import (      # noqa: F401
    HardwareConfig,
    DeviceConfig,
    detect_hardware,
    setup_devices,
    get_active_config,
)

__all__ = [
    # Bootstrap
    "RuntimeConfig",
    "bootstrap",
    "bootstrap_from_yaml_config",
    "get_runtime_config",
    # Backend
    "get_backend",
    "supports_float64",
    "enable_x64",
    "is_x64_enabled",
    "require_x64",
    "check_spectral_backend",
    "configure_backend",
    # Precision
    "PrecisionPolicy",
    "resolve_precision",
    "apply_precision",
    "get_policy",
    "set_policy",
    "cast",
    "const",
    # Devices
    "HardwareConfig",
    "DeviceConfig",
    "detect_hardware",
    "setup_devices",
    "get_active_config",
]
