"""Distributed reduction operations for multi-node MPI.

Provides MPI-aware global sums and maxima that combine local
partial results from each MPI rank.

These functions are only called when the halo backend is set to
``"mpi"``; see :func:`legoesm.grids.halo.set_halo_backend`.

``mpi4jax`` is an optional dependency imported lazily.
"""

from __future__ import annotations

import importlib.util
import os
import re
import warnings

import jax


_TESTED_JAX_MIN = (0, 8, 0)
_TESTED_JAX_MAX_EXCL = (0, 10, 0)
_TESTED_MPI4JAX_MIN = (0, 8, 0)
_TESTED_MPI4JAX_MAX_EXCL = (0, 9, 0)


def _parse_version_triplet(version: str) -> tuple[int, int, int]:
    """Parse a version string into (major, minor, patch) ints."""
    parts = [int(token) for token in re.findall(r"\d+", version)]
    if not parts:
        return (0, 0, 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _format_range(low: tuple[int, int, int], high_excl: tuple[int, int, int]) -> str:
    """Format a half-open version range as text."""
    return (
        f">={low[0]}.{low[1]}.{low[2]}, "
        f"<{high_excl[0]}.{high_excl[1]}.{high_excl[2]}"
    )


def _env_flag_true(name: str) -> bool:
    """Interpret common truthy env values."""
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_mpi_runtime_versions(
    jax_version: str,
    mpi4jax_version: str,
    *,
    strict: bool = False,
) -> None:
    """Validate JAX/mpi4jax runtime compatibility.

    We hard-fail on clearly unsupported mpi4jax versions and warn (or fail in
    strict mode) when versions fall outside the currently tested range.
    """
    jax_triplet = _parse_version_triplet(jax_version)
    mpi4jax_triplet = _parse_version_triplet(mpi4jax_version)

    if mpi4jax_triplet < _TESTED_MPI4JAX_MIN:
        raise RuntimeError(
            "legoESM distributed runtime requires mpi4jax "
            f"{_format_range(_TESTED_MPI4JAX_MIN, _TESTED_MPI4JAX_MAX_EXCL)} "
            f"because older versions use incompatible token semantics. "
            f"Detected mpi4jax=={mpi4jax_version}. "
            f"Fix: pip install 'mpi4jax>=0.8,<0.9'",
        )

    in_tested_jax = _TESTED_JAX_MIN <= jax_triplet < _TESTED_JAX_MAX_EXCL
    in_tested_mpi4jax = _TESTED_MPI4JAX_MIN <= mpi4jax_triplet < _TESTED_MPI4JAX_MAX_EXCL
    if in_tested_jax and in_tested_mpi4jax:
        return

    parts = []
    if not in_tested_jax:
        parts.append(
            f"jax=={jax_version} (tested "
            f"{_format_range(_TESTED_JAX_MIN, _TESTED_JAX_MAX_EXCL)})"
        )
    if not in_tested_mpi4jax:
        parts.append(
            f"mpi4jax=={mpi4jax_version} (tested "
            f"{_format_range(_TESTED_MPI4JAX_MIN, _TESTED_MPI4JAX_MAX_EXCL)})"
        )
    msg = (
        "Detected versions outside legoESM's tested MPI range: "
        + ", ".join(parts)
        + ". MPI execution may fail or produce incorrect results. "
        "Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, "
        "or install tested versions: pip install 'mpi4jax>=0.8,<0.9'"
    )
    if strict:
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _require_mpi_stack():
    """Return (mpi4jax, MPI) or raise a clear ImportError."""
    missing = []
    if importlib.util.find_spec("mpi4jax") is None:
        missing.append("mpi4jax")
    if importlib.util.find_spec("mpi4py") is None:
        missing.append("mpi4py")
    if missing:
        raise ImportError(
            "MPI reductions require optional dependencies "
            f"{', '.join(missing)}. Install them and run under an MPI launcher.",
        )

    import mpi4jax
    from mpi4py import MPI
    _validate_mpi_runtime_versions(
        jax.__version__,
        mpi4jax.__version__,
        strict=_env_flag_true("LEGOESM_MPI_STRICT_COMPAT"),
    )
    return mpi4jax, MPI


def _mpi4jax_array_result(result):
    """Return the array payload from mpi4jax return values.

    mpi4jax<0.8 commonly returned ``(array, token)`` while mpi4jax>=0.8
    returns the array directly with automatic token management.
    """
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("mpi4jax operation returned an empty tuple.")
        return result[0]
    return result


def global_sum_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global sum across all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial sum.

    Returns
    -------
    jax.Array
        The global sum across all processes.
    """
    mpi4jax, MPI = _require_mpi_stack()

    global_val = _mpi4jax_array_result(
        mpi4jax.allreduce(local_value, op=MPI.SUM, comm=MPI.COMM_WORLD),
    )
    return global_val


def global_max_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global maximum across all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial maximum.

    Returns
    -------
    jax.Array
        The global maximum across all processes.
    """
    mpi4jax, MPI = _require_mpi_stack()

    global_val = _mpi4jax_array_result(
        mpi4jax.allreduce(local_value, op=MPI.MAX, comm=MPI.COMM_WORLD),
    )
    return global_val


def global_min_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global minimum across all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial minimum.

    Returns
    -------
    jax.Array
        The global minimum across all processes.
    """
    mpi4jax, MPI = _require_mpi_stack()

    global_val = _mpi4jax_array_result(
        mpi4jax.allreduce(local_value, op=MPI.MIN, comm=MPI.COMM_WORLD),
    )
    return global_val


def allgather_mpi(local_value: jax.Array) -> jax.Array:
    """Gather arrays from all MPI ranks.

    Parameters
    ----------
    local_value : jax.Array
        Local array to gather.

    Returns
    -------
    jax.Array
        Concatenated array from all processes along a new leading axis.
        Shape: ``(n_processes,) + local_value.shape``.
    """
    mpi4jax, MPI = _require_mpi_stack()

    n_procs = MPI.COMM_WORLD.Get_size()
    recv_shape = (n_procs,) + local_value.shape
    recv_buf = jax.numpy.zeros(recv_shape, dtype=local_value.dtype)

    recv_buf = _mpi4jax_array_result(
        mpi4jax.allgather(local_value, comm=MPI.COMM_WORLD),
    )
    return recv_buf


def batch_allreduce_mpi(
    values: list[jax.Array],
    op: str = "sum",
) -> list[jax.Array]:
    """Batch multiple reductions into a single MPI allreduce call.

    Instead of issuing N separate ``allreduce`` calls (each incurring
    MPI latency), this function packs all values into a single flat
    buffer, performs one ``allreduce``, and unpacks the results.

    Parameters
    ----------
    values : list[jax.Array]
        Local partial values to reduce.  Each element can be a scalar
        or an array of any shape; they need not share the same shape.
    op : str
        Reduction operation: ``"sum"`` or ``"max"``.

    Returns
    -------
    list[jax.Array]
        Global reduced values, one per input, with original shapes and
        dtypes restored.

    Examples
    --------
    >>> E, Z, M = batch_allreduce_mpi([E_local, Z_local, M_local])
    >>> v_max, T_max = batch_allreduce_mpi([v_local, T_local], op="max")
    """
    if not values:
        return []

    mpi4jax, MPI = _require_mpi_stack()

    mpi_op_map = {"sum": MPI.SUM, "max": MPI.MAX, "min": MPI.MIN}
    if op not in mpi_op_map:
        raise ValueError(
            f"Unsupported op={op!r}. Choose from {list(mpi_op_map)}."
        )
    mpi_op = mpi_op_map[op]

    # Promote all values to a common dtype for packing.
    import jax.numpy as jnp

    dtypes = [v.dtype for v in values]
    common_dtype = jnp.result_type(*dtypes)
    promoted = [v.astype(common_dtype) for v in values]

    # Record shapes and sizes for unpacking.
    shapes = [v.shape for v in values]
    sizes = [int(v.size) for v in promoted]

    # Pack into a single flat buffer.
    flat_parts = [v.reshape(-1) for v in promoted]
    packed = jnp.concatenate(flat_parts, axis=0)

    # Single MPI allreduce.
    global_packed = _mpi4jax_array_result(
        mpi4jax.allreduce(packed, op=mpi_op, comm=MPI.COMM_WORLD),
    )

    # Unpack and restore original shapes and dtypes.
    results = []
    offset = 0
    for i, (shape, size) in enumerate(zip(shapes, sizes)):
        chunk = global_packed[offset : offset + size].reshape(shape)
        # Cast back to original dtype if it differs from the common one.
        if dtypes[i] != common_dtype:
            chunk = chunk.astype(dtypes[i])
        results.append(chunk)
        offset += size

    return results


def broadcast_mpi(value: jax.Array, root: int = 0) -> jax.Array:
    """Broadcast an array from one rank to all others.

    Parameters
    ----------
    value : jax.Array
        Array to broadcast (only meaningful on ``root``).
    root : int
        Rank that broadcasts.

    Returns
    -------
    jax.Array
        The broadcast value on all ranks.
    """
    mpi4jax, MPI = _require_mpi_stack()

    result = _mpi4jax_array_result(
        mpi4jax.bcast(value, root=root, comm=MPI.COMM_WORLD),
    )
    return result
