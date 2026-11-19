"""Bind a standalone NEMO-fidelity entry point to its owning checkout."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PACKAGE_ROOTS = (
    "packages/core",
    "packages/ocean",
    "packages/atmosphere",
    "packages/coupler",
    "packages/ice",
    "packages/land",
    "packages/ml",
    "packages/tools",
    "src",
)

# JAX 0.10 reads JAX_PLATFORMS=gpu as "cuda or rocm" and refuses to start when ROCm is
# absent; the CUDA plugin's platform name is "cuda", and jax.default_backend() then
# still reports "gpu".
JAX_PLATFORMS_FOR = {"cpu": "cpu", "gpu": "cuda"}


def bootstrap_checkout(
        script: str | Path, *, jax_platform: str = "cpu",
        gpu_device: str | None = None, enforce_x64: bool = True,
) -> dict[str, str]:
    """Select JAX, put the owning checkout first, and verify both choices.

    Importing ``legoesm.ocean`` imports JAX.  Platform selection therefore
    belongs here, before the checkout verification import, rather than later
    in an entry point's model code.  ``enforce_x64`` refuses a process whose
    JAX was already imported in float32; modules that bootstrap at import
    time pass ``__name__ == "__main__"`` so that importing them as a library
    (pytest collection) never depends on which module imported JAX first.
    """
    if jax_platform not in ("cpu", "gpu"):
        raise RuntimeError(
            f"explicit JAX platform must be cpu or gpu, got {jax_platform!r}")
    if jax_platform == "gpu":
        if gpu_device is None or not gpu_device.isdigit():
            raise RuntimeError(
                "GPU entry point requires an explicit non-negative physical "
                "GPU index")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    elif gpu_device is not None:
        raise RuntimeError("gpu_device is valid only for the gpu platform")
    os.environ["JAX_PLATFORMS"] = JAX_PLATFORMS_FOR[jax_platform]
    # Every entry point bound here builds the float64 SWG card.  Without x64
    # the card's own geometry check refuses a correct input with a misleading
    # "grid differs" message, so float64 is selected here, before JAX is
    # imported, instead of being inherited from the caller's shell.
    os.environ["JAX_ENABLE_X64"] = "1"

    path = Path(script).resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot locate the Git checkout owning {path}: {error}"
        ) from error
    roots = [str(root / relative) for relative in _PACKAGE_ROOTS]
    sys.path[:0] = roots
    from legoesm.ocean.fidelity.provenance import require_checkout_import

    result = require_checkout_import(root)
    import jax

    if enforce_x64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX x64 is disabled: JAX was imported with float32 before this "
            "entry point selected float64; run the tool as its own process")
    actual_platform = jax.default_backend()
    if actual_platform != jax_platform:
        raise RuntimeError(
            f"JAX platform mismatch: requested {jax_platform}, selected "
            f"{actual_platform}")
    result.update({
        "jax_platform": actual_platform,
        "physical_gpu": gpu_device,
    })
    print(
        f"JAX_PLATFORM_OK requested={jax_platform} selected={actual_platform} "
        f"physical_gpu={gpu_device if gpu_device is not None else 'NONE'}",
        flush=True,
    )
    return result
