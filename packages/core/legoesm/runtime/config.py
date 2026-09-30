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
        Lowercase backend name (``"cpu"``, ``"gpu"``, ``"tpu"``, ``"mps"``).
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
    distributed_mode: str = "mpi",
    grid_type: str = "cubed_sphere",
    configure_xla: bool = True,
    allow_level_fallback: bool = False,
    grid_n: int | None = None,
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
        Initialise multi-process execution.
    distributed_mode : str
        ``"mpi"`` (default) — mpi4jax halo backend (replicated
        cubed-sphere dynamics / lat-lon band / Voronoi partition).
        ``"spmd"`` — multi-controller ``jax.distributed``: every
        launched process joins ONE program and the device mesh spans
        the GLOBAL device set (true cubed-sphere domain decomposition).
        Cubed-sphere only; mpi4jax is never armed in this mode (the
        two collective stacks deadlock when mixed).
    grid_type : str
        ``"cubed_sphere"``, ``"latlon"``, or ``"spectral"``.
    configure_xla : bool
        Apply per-backend XLA flags.
    allow_level_fallback : bool
        Issue #273 follow-up.  When True and ``grid_type='cubed_sphere'``
        and ``n_devices`` fails face-sharding divisibility (e.g. 4),
        route through the level-parallel cubed-sphere mesh instead
        of clamping to the nearest face-compatible count.  Default
        ``False`` preserves the legacy clamp-or-raise behavior.

    Returns
    -------
    RuntimeConfig
    """
    global _active

    if distributed_mode not in ("mpi", "spmd"):
        raise ValueError(
            f"bootstrap: distributed_mode must be 'mpi' or 'spmd', "
            f"got {distributed_mode!r}"
        )

    # 0. Multi-controller federation (spmd mode) ------------------------------
    # ``jax.distributed.initialize()`` must run before the BACKEND CLIENT
    # is instantiated — i.e. before the first ``jax.devices()`` /
    # ``device_put`` / trace — so every launched process joins one
    # program (the --cs-spmd bench contract).  Plain ``import jax`` does
    # NOT create the client, so module-level jax imports elsewhere are
    # fine; the hazards are device queries before this point.  Steps 1-4
    # below are the first client-touching code on the bootstrap path.
    # Launcher-agnostic process count: SLURM (srun) or OpenMPI (mpirun)
    # — gating on SLURM_NTASKS alone would silently skip initialize()
    # under mpirun and leave N independent local meshes (replicated-
    # serial numbers; bench gate run_cpu_mpi_scaling.py).
    if distributed and distributed_mode == "spmd":
        if grid_type != "cubed_sphere":
            raise ValueError(
                "bootstrap: distributed_mode='spmd' supports only "
                f"grid_type='cubed_sphere' (got {grid_type!r}); the "
                "lat-lon/MPAS distributed paths arm the mpi4jax halo "
                "backend, which must never coexist with "
                "jax.distributed collectives in one program."
            )
        import os as _os
        import jax as _jax

        # STEP-scoped variables first (codex CRITICAL): inside an sbatch
        # allocation with --ntasks=N, an inner ``srun -n 1`` still sees
        # the allocation-wide SLURM_NTASKS=N — keying on it would call
        # initialize() in a 1-process step and hang waiting for N-way
        # federation.  SLURM_STEP_NUM_TASKS / OMPI_COMM_WORLD_SIZE /
        # PMI_SIZE describe the ACTIVE step; SLURM_NTASKS is the
        # last-resort fallback (plain sbatch script body, no srun).
        _nproc = 1
        for _var in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE",
                     "PMI_SIZE", "SLURM_NTASKS"):
            _val = _os.environ.get(_var)
            if _val:
                _nproc = int(_val)
                break
        if _nproc > 1:
            # Idempotent: an OUTER bootstrap (parallel.early_init /
            # initialize_jax_distributed_multiprocess in a launcher
            # script) may already have federated the processes — the
            # supported check is jax.distributed.is_initialized()
            # (cs_spmd smoke, job 8684810; the old "already"-substring
            # guard missed the post-backend-init raise).  Bare
            # initialize() auto-detects SLURM / Open MPI only; under
            # PBS + Cray PALS (Derecho mpiexec) the helper routes to the
            # mpi4py bootstrap (plain-MPI rank/coordinator discovery only
            # — mpi4jax is never armed in spmd mode).
            if not _jax.distributed.is_initialized():
                from legoesm.parallel.early_init import (
                    init_jax_distributed_with_fallback,
                )
                init_jax_distributed_with_fallback()
            if _jax.process_count() != _nproc:
                raise RuntimeError(
                    f"bootstrap: distributed_mode='spmd' launched with "
                    f"{_nproc} processes but jax.process_count()="
                    f"{_jax.process_count()} — jax.distributed did not "
                    "federate them (unsupported launcher?).  Refusing "
                    "to run N independent replicated-serial programs."
                )

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
        # Infer: fp64 stores state in float64, and 'mixed' keeps float64
        # accumulate/control roles — both are meaningless without x64, which
        # would silently demote them to float32 (#1675; the #1665 interim
        # refused 'mixed' here instead).
        need_x64 = precision.strip().lower() in (
            "fp64", "float64", "mixed", "mixed_fp64_storage")

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
        distributed_mode=distributed_mode,
        grid_type=grid_type,
        allow_level_fallback=allow_level_fallback,
        grid_n=grid_n,
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


def precision_mode_from_yaml_config(config) -> str:
    """Precision mode a YAML ``Config`` asks for.

    An explicit ``hardware.precision.mode`` wins; otherwise the per-component
    ``hardware.precision.dynamics`` / ``conservation`` keys decide (fp64
    dynamics -> ``fp64``; fp64 conservation alone -> ``mixed``); nothing set
    gives ``fp32``.
    """
    explicit_mode = config.get("hardware.precision.mode", None)
    if explicit_mode is not None:
        return str(explicit_mode).strip().lower()
    dynamics_prec = config.get("hardware.precision.dynamics", None)
    if dynamics_prec is not None and str(dynamics_prec).strip().lower() in (
            "float64", "fp64", "double"):
        return "fp64"
    cons = config.get("hardware.precision.conservation", None)
    if cons is not None and str(cons).strip().lower() in ("float64", "fp64"):
        return "mixed"
    return "fp32"


def bootstrap_from_yaml_config(config) -> RuntimeConfig:
    """Bootstrap from a legoESM YAML ``Config`` object.

    Reads ``hardware.precision.*``, ``hardware.parallelism.*``, and
    ``hardware.devices`` keys and delegates to :func:`bootstrap`.

    This replaces the old ``core.hardware.apply_hardware_config``.
    """
    precision = precision_mode_from_yaml_config(config)

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

    # Issue #273 follow-up: opt-in level-fallback when the YAML config
    # requests it (``hardware.parallelism.allow_level_fallback: true``
    # or the convenience alias ``shard_radiation_columns: true`` which
    # implies a non-face-divisible device count is expected).
    allow_level_fallback = bool(
        config.get("hardware.parallelism.allow_level_fallback", False)
    )

    return bootstrap(
        precision=precision,
        backend=backend,
        n_devices=n_devices,
        distributed=distributed,
        grid_type=grid_type,
        allow_level_fallback=allow_level_fallback,
    )
