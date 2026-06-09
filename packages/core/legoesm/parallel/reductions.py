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

from legoesm.parallel.profiling import mpi_timer


_TESTED_JAX_MIN = (0, 8, 0)
_TESTED_JAX_MAX_EXCL = (0, 10, 0)
_TESTED_MPI4JAX_MIN = (0, 8, 0)
_TESTED_MPI4JAX_MAX_EXCL = (0, 9, 0)

# mpi4jax 0.8.x uses the deprecated API_VERSION_STATUS_RETURNING custom-call
# convention removed in JAX 0.10.  Until mpi4jax ships an FFI-based release,
# pin JAX < 0.10 for MPI workloads.  The warning from XLA is cosmetic for now
# (the API still functions) but will become a hard error once JAX 0.10 ships.
_MPI4JAX_FFI_MIGRATION_NOTE = (
    "mpi4jax 0.8.x uses a custom-call API deprecated in JAX 0.9 and removed "
    "in JAX 0.10.  Monitor https://github.com/mpi4jax/mpi4jax for an FFI-based "
    "release.  Until then, pin JAX < 0.10 for MPI workloads."
)


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


def _check_mpi4jax_api_version_deprecation() -> str | None:
    """Detect the mpi4jax API_VERSION_STATUS_RETURNING deprecation.

    mpi4jax >= 0.8 changed the custom-call API: operations return arrays
    directly instead of ``(array, token)`` tuples.  When running an older
    mpi4jax against a newer JAX that has dropped the legacy token-based
    custom call path, mpi4jax emits ``API_VERSION_STATUS_RETURNING``
    deprecation warnings and may fail at runtime.

    Returns a diagnostic message if the deprecation is detected, or
    ``None`` if everything looks compatible.
    """
    try:
        import mpi4jax
    except ImportError:
        return None

    # Check for the legacy API marker.  mpi4jax < 0.8 exposed
    # ``API_VERSION_STATUS_RETURNING`` in its XLA bridge layer.
    # If the attribute exists, the installed version uses the deprecated path.
    api_version = getattr(
        getattr(mpi4jax, "_src", None),
        "API_VERSION_STATUS_RETURNING",
        None,
    )
    if api_version is not None:
        return (
            "mpi4jax is using the deprecated API_VERSION_STATUS_RETURNING "
            "custom-call interface, which is incompatible with modern JAX. "
            "Upgrade mpi4jax: pip install 'mpi4jax>=0.8,<0.9'"
        )

    # Also check via XLA custom call registration if available.
    xla_bridge = getattr(getattr(mpi4jax, "_src", None), "xla_bridge", None)
    if xla_bridge is not None:
        for attr_name in dir(xla_bridge):
            if "STATUS_RETURNING" in attr_name.upper():
                return (
                    f"mpi4jax xla_bridge uses deprecated attribute "
                    f"'{attr_name}', indicating an incompatible custom-call "
                    "API. Upgrade mpi4jax: pip install 'mpi4jax>=0.8,<0.9'"
                )

    return None


def _validate_mpi_runtime_versions(
    jax_version: str,
    mpi4jax_version: str,
    *,
    strict: bool = False,
) -> None:
    """Validate JAX/mpi4jax runtime compatibility.

    We hard-fail on clearly unsupported mpi4jax versions and warn (or fail in
    strict mode) when versions fall outside the currently tested range.
    Additionally detects the ``API_VERSION_STATUS_RETURNING`` deprecation
    which indicates an incompatible custom-call API between mpi4jax and JAX.
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

    # Detect API_VERSION_STATUS_RETURNING deprecation (fail-fast).
    api_deprecation_msg = _check_mpi4jax_api_version_deprecation()
    if api_deprecation_msg is not None:
        raise RuntimeError(
            f"Incompatible mpi4jax/JAX combination detected: "
            f"{api_deprecation_msg} "
            f"(jax=={jax_version}, mpi4jax=={mpi4jax_version})"
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
        + _MPI4JAX_FFI_MIGRATION_NOTE + " "
        "Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, "
        "or install tested versions: pip install 'mpi4jax>=0.8,<0.9'"
    )
    if strict:
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def mpi_stack_outside_tested_range() -> bool:
    """``True`` if the installed jax/mpi4jax fall outside legoESM's tested MPI
    range, so distributed *numerics* may be unreliable.

    Halo exchange (point-to-point ``sendrecv``) stays bit-correct across the
    range, but the global-allreduce path (mass fixer, ``global_sum_mpi``) can
    drift ~1e-9 vs serial under an incompatible custom-call ABI — enough to
    break the tight serial-vs-MPI equivalence pins.  Tests that assert that
    equivalence use this to ``xfail`` on an incompatible upstream stack (e.g.
    jax>=0.10.1, for which no mpi4jax release exists yet) while still REQUIRING
    a pass once a tested stack is installed.  See
    :func:`_validate_mpi_runtime_versions`.  Returns ``True`` if either package
    is missing (nothing to run).

    Reads the mpi4jax version from package METADATA rather than importing the
    module: this runs at pytest-collection time (an ``xfail`` condition), and
    importing mpi4jax would initialise the MPI stack as a side effect — which
    can emit OpenMPI bind/runtime errors on a machine not launched under
    ``mpirun`` (codex review P2).  ``jax`` is already imported by this module.
    """
    import importlib.metadata as _md
    import importlib.util as _ilu
    if _ilu.find_spec("mpi4jax") is None:
        return True
    try:
        mpi4jax_version = _md.version("mpi4jax")
    except _md.PackageNotFoundError:
        return True
    jt = _parse_version_triplet(jax.__version__)
    mt = _parse_version_triplet(mpi4jax_version)
    in_jax = _TESTED_JAX_MIN <= jt < _TESTED_JAX_MAX_EXCL
    in_mpi4jax = _TESTED_MPI4JAX_MIN <= mt < _TESTED_MPI4JAX_MAX_EXCL
    return not (in_jax and in_mpi4jax)


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


def global_sum_mpi(local_value: jax.Array, comm=None) -> jax.Array:
    """Compute a global sum across all MPI ranks.

    **Differentiable**: uses ``allreduce(SUM)`` which has full JVP and
    VJP support in mpi4jax.  Safe to use inside ``jax.grad``.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial sum.
    comm : mpi4py communicator, optional
        Communicator to reduce over. Defaults to ``MPI.COMM_WORLD``. Callers on a
        SUB-communicator (e.g. a plane-LES layout whose distributed FFT uses
        ``layout.comm``) MUST pass that same communicator — otherwise the reduction
        spans the wrong rank set and can deadlock or mix unrelated ranks.
    """
    mpi4jax, MPI = _require_mpi_stack()
    if comm is None:
        comm = MPI.COMM_WORLD

    with mpi_timer("global_sum_mpi"):
        global_val = _mpi4jax_array_result(
            mpi4jax.allreduce(local_value, op=MPI.SUM, comm=comm),
        )
    return global_val


def is_multi_process() -> bool:
    """Whether reductions must cross process/rank boundaries.

    ``jax.process_count() > 1`` covers JAX multi-host runs; ``_is_distributed()``
    covers the mpi4jax single-host-multi-rank path where ``process_count`` stays
    1.  Either condition means a local partial sum must be all-reduced to obtain
    the global value.  Canonical home (#177) for the predicate the ocean
    conservation fixers and the eta-floor mass redistribution previously each
    re-implemented identically.
    """
    # Function-scope import: ``core.operators`` imports ``global_sum_mpi`` from
    # this module (function-scope), so importing ``_is_distributed`` at module
    # top level would risk an operators<->reductions import cycle.
    from legoesm.core.operators import _is_distributed
    if jax.process_count() > 1:
        return True
    return _is_distributed()


def global_sum_if_distributed(local_value: jax.Array) -> jax.Array:
    """Global SUM across processes when distributed, else identity.

    Returns ``global_sum_mpi(local_value)`` under multi-process JAX or the
    MPI/sharded distribution flag (see :func:`is_multi_process`); otherwise
    returns ``local_value`` unchanged so single-rank runs pay no reduction.

    **Differentiable**: built on ``global_sum_mpi`` (allreduce SUM) which carries
    a full VJP — safe inside ``jax.grad`` (cf. the halo-exchange ``custom_vjp``
    notes).  Single canonical MPI-aware reduction (#177) shared by
    ``ocean.conservation_mpas`` and ``ocean.dynamics.eta_floor``.
    """
    if is_multi_process():
        return global_sum_mpi(local_value)
    return local_value


def global_max_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global maximum across all MPI ranks.

    **Not differentiable**: ``allreduce(MAX)`` has no meaningful gradient.
    Use only in diagnostics, logging, or CFL monitoring — never in a
    loss function or inside ``jax.grad``.

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

    with mpi_timer("global_max_mpi"):
        global_val = _mpi4jax_array_result(
            mpi4jax.allreduce(local_value, op=MPI.MAX, comm=MPI.COMM_WORLD),
        )
    return global_val


def global_min_mpi(local_value: jax.Array) -> jax.Array:
    """Compute a global minimum across all MPI ranks.

    **Not differentiable**: ``allreduce(MIN)`` has no meaningful gradient.
    Use only in diagnostics, logging, or CFL monitoring — never in a
    loss function or inside ``jax.grad``.

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

    with mpi_timer("global_min_mpi"):
        global_val = _mpi4jax_array_result(
            mpi4jax.allreduce(local_value, op=MPI.MIN, comm=MPI.COMM_WORLD),
        )
    return global_val


def allgather_mpi(local_value: jax.Array) -> jax.Array:
    """Gather arrays from all MPI ranks.

    **Not differentiable**: ``mpi4jax.allgather`` has no JVP or VJP
    rules.  Use only for I/O, checkpoint gathering, and diagnostics
    — never inside ``jax.grad``.

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

    with mpi_timer("allgather_mpi"):
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
    with mpi_timer("batch_allreduce_mpi"):
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

    **Not differentiable**: ``mpi4jax.bcast`` has no JVP or VJP rules.
    Use only for initialization and I/O — never inside ``jax.grad``.

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
