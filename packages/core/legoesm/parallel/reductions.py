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

# Second compatible generation — the FFI-based mpi4jax line.  mpi4jax 0.8.x used a
# custom-call API removed in jax 0.10; mpi4jax >= 0.9 ships FFI-based custom calls that
# REQUIRE jax >= 0.10 (the INVERSE pairing of the legacy [_TESTED_*] range).  This
# generation is VERIFIED-working: the distributed compare-reanalysis suite (serial-vs-MPI
# equivalence + differentiability, 17 tests under `mpirun -np 2`) passes on jax==0.10.0 +
# mpi4jax==0.9.0.post1 (iter 333).  Accept it so the operator is not false-warned off the
# working HPC stack.  Bounds are capped at the verified versions' next minor (conservative:
# a future jax 0.11 / mpi4jax 0.10 warns again until re-verified).
# Both generations (and the CROSS-pairing rejection) are now folded into the single
# `_stack_in_tested_generation` regime predicate, shared by BOTH the runtime preflight and
# the test xfail gate `mpi_stack_outside_tested_range` — so the two can never disagree
# (the earlier "conservative" gate xfailed the verified FFI stack the runtime already
# accepted, which read as a "broken MPI stack"; verified jax 0.10.1 + mpi4jax 0.9.0.post1
# 2026-07-13: cube 44/44 + voronoi 81 pass on `mpirun -np 2`).
_FFI_JAX_MIN = (0, 10, 0)
_FFI_JAX_MAX_EXCL = (0, 11, 0)
_FFI_MPI4JAX_MIN = (0, 9, 0)
_FFI_MPI4JAX_MAX_EXCL = (0, 10, 0)

# Two compatible generations (each package must be paired WITHIN one generation):
# legacy custom-call (mpi4jax 0.8.x + jax 0.8-0.9) and FFI-based (mpi4jax 0.9.x + jax
# 0.10.x).  mpi4jax 0.8.x's custom-call API was removed in jax 0.10; mpi4jax 0.9.x ships
# the FFI-based replacement that REQUIRES jax >= 0.10 (verified-working, iter 333).  A
# CROSS pairing is the genuinely-incompatible case.
_MPI4JAX_FFI_MIGRATION_NOTE = (
    "legoESM supports two compatible generations paired TOGETHER: legacy "
    "(jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is "
    "incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the "
    "mpi4jax 0.9 FFI API needs jax >= 0.10)."
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


def _stack_in_tested_generation(
    jax_triplet: tuple[int, int, int],
    mpi4jax_triplet: tuple[int, int, int],
) -> bool:
    """``True`` if (jax, mpi4jax) form a VERIFIED-compatible generation.

    Two generations, each of which must be paired WITHIN itself:
      * legacy custom-call: jax 0.8-0.9 + mpi4jax 0.8.x, and
      * FFI-based:          jax 0.10.x + mpi4jax 0.9.x (verified-working, iter 333).
    A CROSS pairing (e.g. jax 0.10 + mpi4jax 0.8, whose custom-call API was
    removed in jax 0.10; or jax 0.8 + mpi4jax 0.9, whose FFI API needs jax>=0.10)
    is the genuinely-incompatible case and returns ``False``.

    The single source of truth for BOTH the production preflight
    (:func:`_validate_mpi_runtime_versions`) and the test xfail gate
    (:func:`mpi_stack_outside_tested_range`) — so a stack the runtime accepts
    can never be simultaneously xfailed by the suite.
    """
    legacy = (
        _TESTED_JAX_MIN <= jax_triplet < _TESTED_JAX_MAX_EXCL
        and _TESTED_MPI4JAX_MIN <= mpi4jax_triplet < _TESTED_MPI4JAX_MAX_EXCL
    )
    ffi = (
        _FFI_JAX_MIN <= jax_triplet < _FFI_JAX_MAX_EXCL
        and _FFI_MPI4JAX_MIN <= mpi4jax_triplet < _FFI_MPI4JAX_MAX_EXCL
    )
    return legacy or ffi


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
            "Upgrade mpi4jax: pip install 'mpi4jax>=0.9,<0.10'"
        )

    # Also check via XLA custom call registration if available.
    xla_bridge = getattr(getattr(mpi4jax, "_src", None), "xla_bridge", None)
    if xla_bridge is not None:
        for attr_name in dir(xla_bridge):
            if "STATUS_RETURNING" in attr_name.upper():
                return (
                    f"mpi4jax xla_bridge uses deprecated attribute "
                    f"'{attr_name}', indicating an incompatible custom-call "
                    "API. Upgrade mpi4jax: pip install 'mpi4jax>=0.9,<0.10'"
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
            f"Fix: pip install 'mpi4jax>=0.9,<0.10'",
        )

    # Detect API_VERSION_STATUS_RETURNING deprecation (fail-fast).
    api_deprecation_msg = _check_mpi4jax_api_version_deprecation()
    if api_deprecation_msg is not None:
        raise RuntimeError(
            f"Incompatible mpi4jax/JAX combination detected: "
            f"{api_deprecation_msg} "
            f"(jax=={jax_version}, mpi4jax=={mpi4jax_version})"
        )

    # Accept EITHER verified generation (legacy or FFI), paired within itself;
    # a cross pairing falls through to the warning below.  ``in_tested_*`` (the
    # LEGACY range membership) is retained only to phrase that warning.
    if _stack_in_tested_generation(jax_triplet, mpi4jax_triplet):
        return
    in_tested_jax = _TESTED_JAX_MIN <= jax_triplet < _TESTED_JAX_MAX_EXCL
    in_tested_mpi4jax = _TESTED_MPI4JAX_MIN <= mpi4jax_triplet < _TESTED_MPI4JAX_MAX_EXCL

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
    # Generation-aware remediation: guide the user to the CONSISTENT pairing for THEIR
    # jax era (the legacy `pip install 'mpi4jax>=0.8,<0.9'` was wrong for a jax>=0.10 user,
    # who needs the FFI line — iter 334).
    if jax_triplet >= _FFI_JAX_MIN:
        fix = (
            "for this jax (>=0.10) install the FFI mpi4jax line: "
            "pip install 'mpi4jax>=0.9,<0.10'"
        )
    else:
        fix = (
            "for this jax (<0.10) install the legacy mpi4jax line: "
            "pip install 'mpi4jax>=0.8,<0.9'"
        )
    msg = (
        "Detected versions outside legoESM's tested MPI range: "
        + ", ".join(parts)
        + ". MPI execution may fail or produce incorrect results. "
        + _MPI4JAX_FFI_MIGRATION_NOTE + " "
        "Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or "
        + fix + "."
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
    equivalence use this to ``xfail`` on an incompatible upstream stack (a CROSS
    pairing, or a jax/mpi4jax beyond the verified generations) while still
    REQUIRING a pass once a tested stack is installed.  Shares
    :func:`_stack_in_tested_generation` with :func:`_validate_mpi_runtime_versions`,
    so a stack the runtime accepts is never simultaneously xfailed here (the FFI
    generation jax 0.10.x + mpi4jax 0.9.x is accepted).  Returns ``True`` if
    either package is missing (nothing to run).

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
    return not _stack_in_tested_generation(jt, mt)


# --- mpi4jax GPU transport (device-direct vs host-staged) -------------------
# mpi4jax decides, once per process, whether a halo ``sendrecv`` hands the
# on-device buffer straight to (GPU-aware) MPI or first copies
# device->host->device.  The switch is the env var ``MPI4JAX_USE_CUDA_MPI``
# (read in mpi4jax's decorators): unset/falsy -> HOST-STAGED (always safe, but
# the per-exchange host round-trip caps multi-GPU scaling); truthy ->
# GPU-DIRECT (fast, but segfaults if mpi4jax has no CUDA extension or the MPI is
# not GPU-aware).  legoESM hands device arrays to ``sendrecv`` unconditionally,
# so on a GPU backend this choice is otherwise INVISIBLE -- a silent host-stage
# looks like "the halo works but doesn't scale".  ``mpi4jax.has_cuda_support()``
# reports whether the CUDA extension was built in; a missing or raising
# predicate is treated as UNPROVEN (we fail closed when GPU-direct is requested).
_GPU_TRANSPORT_BACKENDS = ("gpu", "cuda", "rocm")
_mpi4jax_transport_warned = False

_MPI4JAX_NO_CUDA_EXT_MSG = (
    "MPI4JAX_USE_CUDA_MPI is set (GPU-direct halo exchange requested) on a GPU "
    "backend, but this mpi4jax does not report usable CUDA support "
    "(mpi4jax.has_cuda_support() returned False or could not be called): handing "
    "an on-device sendrecv buffer to MPI would error or segfault. Rebuild "
    "mpi4jax with CUDA against the GPU-aware Cray MPICH "
    "(scripts/cluster/scaling_derecho/README.md), or unset MPI4JAX_USE_CUDA_MPI "
    "to use the (slower) host-staged path."
)
_MPI4JAX_HOST_STAGED_MSG = (
    "GPU backend with a CUDA-capable mpi4jax, but MPI4JAX_USE_CUDA_MPI is unset: "
    "every halo sendrecv copies device->host->device, which caps multi-GPU "
    "scaling. Export MPI4JAX_USE_CUDA_MPI=1 for GPU-direct halo exchange (also "
    "needs MPICH_GPU_SUPPORT_ENABLED=1 and the craype-accel-nvidia80 GTL on "
    "Cray/Slingshot)."
)


def _mpi4jax_transport_action(
    backend: str, use_cuda_mpi: bool, cuda_built: bool | None
) -> str:
    """Decide the halo-transport diagnostic: ``"ok"`` | ``"warn"`` | ``"raise"``.

    Pure (no I/O, no globals) so the policy is unit-testable in isolation.

    * ``"raise"`` -- GPU backend with GPU-direct REQUESTED (``use_cuda_mpi``) but
      CUDA support not POSITIVELY proven (``cuda_built`` is ``False`` *or*
      ``None``/unintrospectable): handing an on-device ``sendrecv`` buffer to MPI
      would error or segfault, so fail CLOSED before the first halo exchange.
    * ``"warn"``  -- GPU backend, the toggle UNSET, and a CUDA-capable mpi4jax:
      the halo silently host-stages (device<->host copy), capping GPU scaling.
    * ``"ok"``    -- CPU/TPU (toggle irrelevant); a proven GPU-direct run; or a
      host-staged run whose build offers no usable CUDA path anyway (nothing
      actionable to say).
    """
    if backend not in _GPU_TRANSPORT_BACKENDS:
        return "ok"
    if use_cuda_mpi:
        # GPU-direct demanded: only a positive proof of CUDA support is safe.
        # ``None`` (predicate missing/raising) is NOT proof -> fail closed.
        return "ok" if cuda_built is True else "raise"
    # Toggle unset -> host-staged (safe). Nudge only when GPU-direct is actually
    # available; stay quiet when it is unavailable or cannot be proven.
    return "warn" if cuda_built is True else "ok"


def check_mpi4jax_transport(mpi4jax) -> None:
    """Surface the mpi4jax GPU halo transport mode.

    Called from every checked entry point that can issue a device-array MPI op:
    :func:`require_mpi_stack` (collectives) and the halo sendrecv choke point
    :func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`.  Trace-safe: pure
    Python, no JAX ops.
    The hard-error condition is RECOMPUTED on every call (never latched) so a
    later ``MPI4JAX_USE_CUDA_MPI`` / backend / build change cannot slip a
    GPU-direct misconfiguration past the preflight; only the advisory
    host-staging warning is rate-limited to once per process.
    """
    global _mpi4jax_transport_warned
    backend = jax.default_backend()
    if backend not in _GPU_TRANSPORT_BACKENDS:
        return  # CPU/TPU: the device-direct toggle is moot; never poke mpi4jax.
    try:
        cuda_built: bool | None = bool(mpi4jax.has_cuda_support())
    except Exception:
        cuda_built = None  # predicate missing/raising -> CUDA support UNPROVEN
    action = _mpi4jax_transport_action(
        backend,
        _env_flag_true("MPI4JAX_USE_CUDA_MPI"),
        cuda_built,
    )
    if action == "raise":
        raise ImportError(_MPI4JAX_NO_CUDA_EXT_MSG)
    if action == "warn" and not _mpi4jax_transport_warned:
        _mpi4jax_transport_warned = True
        warnings.warn(_MPI4JAX_HOST_STAGED_MSG, RuntimeWarning, stacklevel=2)


def require_mpi_stack():
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
    check_mpi4jax_transport(mpi4jax)
    return mpi4jax, MPI


def mpi4jax_array_result(result):
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
    mpi4jax, MPI = require_mpi_stack()
    if comm is None:
        comm = MPI.COMM_WORLD

    with mpi_timer("global_sum_mpi"):
        global_val = mpi4jax_array_result(
            mpi4jax.allreduce(local_value, op=MPI.SUM, comm=comm),
        )
    return global_val


def is_multi_process() -> bool:
    """Whether a local partial sum must be MANUALLY all-reduced via mpi4jax.

    True exactly when an mpi4jax-style decomposition is armed — the MPI halo
    backend (``is_distributed()``) or a Voronoi partition layout — i.e. when
    each rank holds a LOCAL shard as a plain per-process array.  Canonical
    home (#177) for the predicate the ocean conservation fixers and the
    eta-floor mass redistribution previously each re-implemented identically.

    ``jax.process_count() > 1`` (a jax.distributed multi-controller run) is
    deliberately NOT a trigger: mpi4jax is never armed in that mode (mixed
    jax.distributed + mpi4jax stacks deadlock), and no manual reduction is
    needed there — outside ``shard_map`` the state lives in GLOBAL jax.Arrays
    whose ``jnp`` reductions are already global, and inside ``shard_map`` the
    armed "spmd" halo backend routes the same call sites to ``psum`` branches
    checked BEFORE this predicate.  The old ``process_count`` clause sent the
    multi-controller ocean serial-reference/invariant legs into
    ``batch_allreduce_mpi`` — importing mpi4jax into a program where it must
    never run.  Refusal-style guards that need ANY-multi-process semantics
    must additionally check ``jax.process_count()`` themselves (see the
    ``ocean_pe_mpas`` normalize_freshwater guard).
    """
    # Function-scope import: ``core.operators`` imports ``global_sum_mpi`` from
    # this module (function-scope), so importing ``is_distributed`` at module
    # top level would risk an operators<->reductions import cycle.
    from legoesm.core.operators import is_distributed
    if is_distributed():
        return True
    # Voronoi/MPAS cell-partition MPI arms NO halo backend (it carries a
    # partition layout instead), so the two checks above are FALSE there
    # — the distributed-MPAS-PCG triangulation (job 8460616) caught
    # ``_global_dot_batch`` silently skipping its allreduce: every rank
    # exactly solved its own half-system (rel_res 1e-17) while
    # disagreeing globally.  The active layout is the multi-process
    # signal on that path.  NOTE for consumers: Voronoi local arrays
    # carry HALO entities — a partial sum feeding the allreduce must be
    # owned-masked or it double-counts (see
    # ``VoronoiPartitionLayout.owned_mask_cells``).
    from legoesm.parallel.voronoi_mpi import get_active_voronoi_layout
    return get_active_voronoi_layout() is not None


def mpi_world_size() -> int:
    """``MPI_COMM_WORLD`` size if mpi4py is importable, else 1.

    Host-level Python (trace-safe, no JAX ops).  Complements
    :func:`is_multi_process` for FAIL-FAST guards: the Voronoi/MPAS MPI
    path builds a partition layout WITHOUT arming the global halo
    backend, so ``is_distributed()`` stays False there and a guard
    keyed on :func:`is_multi_process` alone never fires (codex review
    2026-06-11 CRITICAL — the MPAS implicit_cn refusal was unreachable
    in exactly the ``mpirun -np N`` scenario it targeted).  A guard
    using ``is_multi_process() or mpi_world_size() > 1`` trips on any
    real multi-rank launch.  Caveat: an ``mpirun`` ensemble of
    INDEPENDENT serial members also trips such guards — that pattern
    is not used in this repo (ensembles batch via vmap).

    Absent (``ImportError``) or unloadable (the loader's ``RuntimeError``,
    e.g. no libmpi in a GPU-only venv) mpi4py returns 1 ONLY when no MPI
    launcher started the process — under launcher evidence
    (OMPI/PMI/PALS/SLURM env) it raises LOUDLY, or every rank of a real
    multi-rank launch would sail past the fail-fast guards keyed on this
    function as "serial" (codex rounds 4-5; the same policy as the bench
    drivers' ``_init_mpi``).
    """
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError) as e:
        launcher = any(
            v in os.environ
            for v in ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK",
                      "PMI_SIZE", "PMI_RANK", "PALS_RANKID",
                      "PALS_LOCAL_SIZE")
        ) or (os.environ.get("SLURM_NTASKS", "1").isdigit()
              and int(os.environ.get("SLURM_NTASKS", "1")) > 1)
        if launcher:
            raise RuntimeError(
                "MPI launcher detected but mpi4py is unavailable/unusable: "
                f"{e}"
            ) from e
        return 1
    return int(MPI.COMM_WORLD.Get_size())


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
    mpi4jax, MPI = require_mpi_stack()

    with mpi_timer("global_max_mpi"):
        global_val = mpi4jax_array_result(
            mpi4jax.allreduce(local_value, op=MPI.MAX, comm=MPI.COMM_WORLD),
        )
    return global_val


def global_min_mpi(local_value: jax.Array, comm=None) -> jax.Array:
    """Compute a global minimum across all MPI ranks.

    **Not differentiable**: ``allreduce(MIN)`` has no meaningful gradient.
    Use only in diagnostics, logging, CFL monitoring, or host-side control
    flow — never in a loss function or inside ``jax.grad``.

    Parameters
    ----------
    local_value : jax.Array
        Scalar (or array) local partial minimum.
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD`` — mirrors :func:`global_sum_mpi` so a
        caller on a subset communicator reduces over ITS group (a hardcoded
        world communicator would hang when nonmembers never enter the
        reduction — codex 2026-08-16, non-finite-guard review, finding 1).

    Returns
    -------
    jax.Array
        The global minimum across the communicator's processes.
    """
    mpi4jax, MPI = require_mpi_stack()

    with mpi_timer("global_min_mpi"):
        global_val = mpi4jax_array_result(
            mpi4jax.allreduce(
                local_value, op=MPI.MIN,
                comm=comm if comm is not None else MPI.COMM_WORLD),
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
    mpi4jax, MPI = require_mpi_stack()

    n_procs = MPI.COMM_WORLD.Get_size()
    recv_shape = (n_procs,) + local_value.shape
    recv_buf = jax.numpy.zeros(recv_shape, dtype=local_value.dtype)

    with mpi_timer("allgather_mpi"):
        recv_buf = mpi4jax_array_result(
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

    mpi4jax, MPI = require_mpi_stack()

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
        global_packed = mpi4jax_array_result(
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


#: Device-shard axes an ocean SPMD step may be sharded over, by lane:
#: ``"lat"`` = lat-lon band lane (``activate_latlon_spmd_halo``), ``"device"``
#: = Voronoi/MPAS lane (``voronoi_spmd_ocean``).  Cube-atm SPMD meshes carry
#: neither and fall through to the MPI / serial gates.
_OCEAN_SPMD_AXES = ("lat", "device")


def spmd_reduce_axis() -> str | None:
    """The mesh axis an in-``shard_map`` ocean reduction must ``psum`` over,
    or ``None`` when no ocean SPMD lane is armed.

    Single canonical gate for the ``"spmd"`` halo-backend branch of every
    ocean reduction site (barotropic PCG dots, eta-floor redistribution) —
    previously each site re-implemented the ``get_halo_backend() == "spmd"``
    + ``"lat" in mesh.axis_names`` check and so silently skipped the Voronoi
    lane's ``"device"`` axis.  Raises when the backend says ``"spmd"`` but no
    mesh is set (an invalid arming reachable only via the raw setter).
    """
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() != "spmd":
        return None
    mesh = get_spmd_mesh()
    if mesh is None:
        raise RuntimeError(
            "halo backend is 'spmd' but no SPMD mesh is set; arm it via "
            "activate_latlon_spmd_halo(mesh) / build_mpas_ocean_spmd_layout.")
    names = tuple(mesh.axis_names)
    for ax in _OCEAN_SPMD_AXES:
        if ax in names:
            return ax
    return None


def batch_psum_spmd(
    values: list[jax.Array],
    axis_name: str | tuple[str, ...],
) -> list[jax.Array]:
    """Batch multiple SPMD reductions into a single ``jax.lax.psum``.

    The single-controller (``shard_map``) analogue of
    :func:`batch_allreduce_mpi`: instead of ``mpi4jax.allreduce`` it sums
    the packed buffer with :func:`jax.lax.psum` over the mesh ``axis_name``
    (the device-shard axis).  Used by the lat-lon band SPMD ocean step
    (route-B, pure-jax multi-GPU — no mpi4jax), where the barotropic PCG's
    inner products are local PARTIAL sums over each device's latitude band
    and must be summed across the ``"lat"`` axis to obtain the global dot.

    MUST be called INSIDE a ``shard_map`` whose mesh carries ``axis_name``
    (``jax.lax.psum`` needs the axis in scope); the lat-lon ocean step
    arms it via :func:`legoesm.parallel.latlon_spmd.activate_latlon_spmd_halo`.

    **Differentiable**: ``jax.lax.psum`` is self-transposing (its VJP is a
    ``psum`` of the cotangents), so unrolling the fixed-M PCG and
    differentiating straight through these reductions is AD-safe — the same
    property ``allreduce(SUM)`` has in the MPI path (CLAUDE.md MPI-AD
    doctrine).

    Parameters
    ----------
    values : list[jax.Array]
        Local partial values (per-device) to sum across ``axis_name``.
        Each may be a scalar or array of any shape; shapes need not match.
    axis_name : str | tuple[str, ...]
        The ``shard_map`` mesh axis (or axes) over which the field is
        sharded — the device dimension the partial sums must combine over.

    Returns
    -------
    list[jax.Array]
        Globally-summed values, original shapes and dtypes restored.
    """
    if not values:
        return []

    import jax.numpy as jnp

    # Pack -> ONE psum -> unpack (mirrors batch_allreduce_mpi so a single
    # collective carries all CG scalars of an iteration).
    dtypes = [v.dtype for v in values]
    common_dtype = jnp.result_type(*dtypes)
    promoted = [v.astype(common_dtype) for v in values]

    shapes = [v.shape for v in values]
    sizes = [int(v.size) for v in promoted]

    flat_parts = [v.reshape(-1) for v in promoted]
    packed = jnp.concatenate(flat_parts, axis=0)

    global_packed = jax.lax.psum(packed, axis_name)

    results = []
    offset = 0
    for i, (shape, size) in enumerate(zip(shapes, sizes)):
        chunk = global_packed[offset : offset + size].reshape(shape)
        if dtypes[i] != common_dtype:
            chunk = chunk.astype(dtypes[i])
        results.append(chunk)
        offset += size

    return results
