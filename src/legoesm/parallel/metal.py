"""Mac Metal GPU + CPU hybrid routing.

Apple Metal GPUs lack float64 and complex128 support, so the
spectral solver must run on CPU while the finite-volume solver
(cubed-sphere, float32) can benefit from Metal acceleration.

This module provides automatic device routing:

- :func:`get_metal_config` — detect Metal and get both devices.
- :func:`to_cpu` / :func:`to_metal` — transfer arrays between devices.
- Model classes use these to route computation transparently.

When not running on Metal (or when Metal is non-functional and has
fallen back to CPU), all functions are no-ops.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.runtime.backend import get_backend, metal_fell_back_to_cpu


class MetalConfig(NamedTuple):
    """Metal device configuration.

    Attributes
    ----------
    metal_device : jax.Device or None
        The Metal GPU device, or ``None`` if not on Metal.
    cpu_device : jax.Device
        The CPU device (always available).
    is_metal : bool
        ``True`` if the default backend is Metal **and** it is functional.
        ``False`` when Metal was detected but fell back to CPU.
    """
    metal_device: jax.Device | None
    cpu_device: jax.Device
    is_metal: bool


def get_metal_config() -> MetalConfig:
    """Detect the Metal backend and resolve CPU/GPU devices.

    If Metal is detected but non-functional (e.g. jax-metal / JAX version
    mismatch), ``is_metal`` will be ``False`` because the runtime has
    already fallen back to CPU.

    Returns
    -------
    MetalConfig
    """
    backend = get_backend()  # triggers health check & fallback
    cpu_device = jax.devices("cpu")[0]

    if backend == "metal":
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
    """Transfer an array to the default Metal device.

    If not on Metal (or Metal fell back to CPU), returns the array unchanged.

    Parameters
    ----------
    array : jax.Array
        Array on any device.

    Returns
    -------
    jax.Array on the Metal device (or unchanged).
    """
    backend = get_backend()
    if backend != "metal":
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

    When Metal has fallen back to CPU, this routes to CPU (not to the
    non-functional Metal device).

    Parameters
    ----------
    pytree
        Any JAX pytree.

    Returns
    -------
    Pytree with all array leaves on the default device.
    """
    # jax.config.jax_default_device is set when Metal fell back to CPU.
    # jax.devices()[0] would still return the Metal device in that case.
    default = getattr(jax.config, "jax_default_device", None)
    if default is None:
        default = jax.devices()[0]
    return jax.device_put(pytree, default)


def is_metal_backend() -> bool:
    """Check if the default JAX backend is Metal **and** functional.

    Returns ``False`` when Metal was detected but fell back to CPU.
    """
    return get_backend() == "metal"


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
    """Wrap a function so that on Metal, inputs are routed to CPU.

    On Metal, float64 and complex128 are unsupported. This decorator
    transfers inputs to CPU, runs the function, and transfers results
    back to the default device.

    Skips redundant transfers when inputs are already on the target
    device (e.g., within a ``lax.scan`` body that already runs on CPU).

    On non-Metal backends this is a no-op wrapper.

    The Metal-vs-non-Metal check is repeated at *call* time as well as
    at decoration time: if Metal silently falls back to CPU (which
    flips ``is_metal_backend()`` to False), the wrapper short-circuits
    so we don't pay the device→device tree-map roundtrip on every call.
    """
    if not is_metal_backend():
        return fn

    _cpu = jax.devices("cpu")[0]

    def wrapper(*args, **kwargs):
        # Re-check at call time — Metal may have fallen back to CPU
        # since decoration; in that case there is nothing to route.
        if not is_metal_backend():
            return fn(*args, **kwargs)
        args_cpu = jax.tree.map(lambda x: _put_if_needed(x, _cpu), args)
        kwargs_cpu = jax.tree.map(lambda x: _put_if_needed(x, _cpu), kwargs)
        result = fn(*args_cpu, **kwargs_cpu)
        return route_to_default(result)

    return wrapper
