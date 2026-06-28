"""Mac Apple GPU (Metal) + CPU hybrid routing.

Apple GPUs (via the ``mps`` backend — jax-mps / MLX) lack float64 and
complex128 support, so the spectral solver must run on CPU while the
finite-volume solver (cubed-sphere, float32) can benefit from GPU
acceleration.

This module provides automatic device routing:

- :func:`get_metal_config` — detect the Apple GPU and get both devices.
- :func:`to_cpu` / :func:`to_metal` — transfer arrays between devices.
- Model classes use these to route computation transparently.

When not running on the Apple GPU (``mps``) backend, all functions are
no-ops.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax

from legoesm.runtime.backend import check_spectral_backend, get_backend


class MetalConfig(NamedTuple):
    """Apple GPU (Metal) device configuration.

    Attributes
    ----------
    metal_device : jax.Device or None
        The Apple GPU device, or ``None`` if not on the ``mps`` backend.
    cpu_device : jax.Device
        The CPU device (always available).
    is_metal : bool
        ``True`` if the default backend is the Apple GPU (``mps``).
    """
    metal_device: jax.Device | None
    cpu_device: jax.Device
    is_metal: bool


def get_metal_config() -> MetalConfig:
    """Detect the Apple GPU (``mps``) backend and resolve CPU/GPU devices.

    Returns
    -------
    MetalConfig
    """
    backend = get_backend()
    cpu_device = jax.devices("cpu")[0]

    if backend == "mps":
        metal_devices = jax.devices()
        metal_device = metal_devices[0] if metal_devices else None
        return MetalConfig(
            metal_device=metal_device,
            cpu_device=cpu_device,
            is_metal=True,
        )

    return MetalConfig(
        metal_device=None,
        cpu_device=cpu_device,
        is_metal=False,
    )


class SpectralDevicePlacement(NamedTuple):
    """Resolved device routing for a spectral model's grid.

    Attributes
    ----------
    grid
        The (possibly CPU-transferred) grid to store on the model.
    use_cpu_for_spectral : bool
        ``True`` only on the Apple GPU (``mps``) backend.
    cpu_device, default_device : jax.Device or None
        Set only when routing is active (``mps``); ``None`` otherwise — matches
        the sentinel convention the spectral dycores already use.
    """

    grid: Any
    use_cpu_for_spectral: bool
    cpu_device: jax.Device | None
    default_device: jax.Device | None


def place_spectral_grid(grid, *, allow_unsupported: bool = False) -> SpectralDevicePlacement:
    """Route a spectral grid to a device that supports float64/complex128.

    On the Apple GPU (``mps``) the grid is transferred to CPU (MLX lacks
    fp64/complex128) and the returned flags let the model run its transforms
    there; on every other backend the grid is returned untouched after
    :func:`legoesm.runtime.backend.check_spectral_backend` verifies fp64
    support (or merely warns with ``allow_unsupported=True``).

    Single home for the constructor block previously copy-pasted across the
    four spectral dycores (atmosphere ``spectral_sw``/``spectral_pe``/
    ``spectral_nh``, ocean ``spectral_ocean_pe``) — behavior-identical to
    those blocks. Must run before any float64 computation on the grid.
    """
    backend = get_backend()
    if backend == "mps":
        cpu_device = jax.devices("cpu")[0]
        return SpectralDevicePlacement(
            grid=jax.device_put(grid, cpu_device),
            use_cpu_for_spectral=True,
            cpu_device=cpu_device,
            default_device=jax.devices()[0],
        )
    check_spectral_backend(allow_unsupported=allow_unsupported)
    return SpectralDevicePlacement(
        grid=grid,
        use_cpu_for_spectral=False,
        cpu_device=None,
        default_device=None,
    )


def to_cpu(array: jax.Array) -> jax.Array:
    """Transfer an array to the CPU device.

    Parameters
    ----------
    array : jax.Array
        Array on any device.

    Returns
    -------
    jax.Array on CPU.
    """
    cpu = jax.devices("cpu")[0]
    return jax.device_put(array, cpu)


def to_metal(array: jax.Array) -> jax.Array:
    """Transfer an array to the default Apple GPU (``mps``) device.

    If not on the ``mps`` backend, returns the array unchanged.

    Parameters
    ----------
    array : jax.Array
        Array on any device.

    Returns
    -------
    jax.Array on the Apple GPU device (or unchanged).
    """
    backend = get_backend()
    if backend != "mps":
        return array
    metal = jax.devices()[0]
    return jax.device_put(array, metal)


def route_to_cpu(pytree):
    """Transfer an entire pytree to CPU.

    Parameters
    ----------
    pytree
        Any JAX pytree.

    Returns
    -------
    Pytree with all array leaves on CPU.
    """
    cpu = jax.devices("cpu")[0]
    return jax.device_put(pytree, cpu)


def route_to_default(pytree):
    """Transfer an entire pytree to the effective default device.

    Parameters
    ----------
    pytree
        Any JAX pytree.

    Returns
    -------
    Pytree with all array leaves on the default device.
    """
    # Honour an explicit ``jax_default_device`` override if one is set;
    # otherwise route to the backend's first device.
    default = getattr(jax.config, "jax_default_device", None)
    if default is None:
        default = jax.devices()[0]
    return jax.device_put(pytree, default)


def is_metal_backend() -> bool:
    """Check if the default JAX backend is the Apple GPU (``mps``)."""
    return get_backend() == "mps"


def _is_on_device(x, device):
    """Check if a JAX array is already on the given device."""
    if not hasattr(x, "devices"):
        return True  # non-array (e.g., python scalar)
    try:
        devs = x.devices()
        return device in devs
    except Exception:
        return False


def _put_if_needed(x, device):
    """Transfer *x* to *device* only if it's not already there."""
    if not hasattr(x, "dtype"):
        return x
    if _is_on_device(x, device):
        return x
    return jax.device_put(x, device)


def ensure_spectral_on_cpu(fn):
    """Wrap a function so that on the Apple GPU (``mps``), inputs are routed to CPU.

    On ``mps``, float64 and complex128 are unsupported. This decorator
    transfers inputs to CPU, runs the function, and transfers results
    back to the default device.

    Skips redundant transfers when inputs are already on the target
    device (e.g., within a ``lax.scan`` body that already runs on CPU).

    On non-``mps`` backends this is a no-op wrapper.

    The mps-vs-non-mps check is repeated at *call* time as well as at
    decoration time so the wrapper short-circuits when there is nothing
    to route, avoiding the device→device tree-map roundtrip on every call.
    """
    if not is_metal_backend():
        return fn

    _cpu = jax.devices("cpu")[0]

    def wrapper(*args, **kwargs):
        # Re-check at call time for robustness; if the backend is no
        # longer mps there is nothing to route.
        if not is_metal_backend():
            return fn(*args, **kwargs)
        args_cpu = jax.tree.map(lambda x: _put_if_needed(x, _cpu), args)
        kwargs_cpu = jax.tree.map(lambda x: _put_if_needed(x, _cpu), kwargs)
        result = fn(*args_cpu, **kwargs_cpu)
        return route_to_default(result)

    return wrapper
