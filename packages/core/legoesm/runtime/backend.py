"""Backend detection, XLA configuration, and X64 policy.

This is the single source of truth for all backend-related queries and
environment setup.  It consolidates logic previously split across
``core.hardware`` and ``parallel.device_config``.

Key design rules:

* **No JAX import at module scope** — JAX is imported lazily inside
  functions so that environment variables (``JAX_PLATFORMS``,
  ``XLA_FLAGS``, etc.) can be set *before* JAX initialises.
* All public helpers are pure queries or idempotent setters.
"""

from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)

# Backends that lack float64/complex128 hardware support.  The Apple GPU
# backend ``mps`` (jax-mps / MLX) is float32-only — MLX has no float64.
# ``metal`` is the legacy Apple-GPU platform string, kept here so the no-f64
# guard never silently passes if a stale config/caller hands it to
# ``supports_float64``/``check_spectral_backend`` (that backend is no longer
# wired in; its Metal GPUs also lacked float64).
_NO_F64_BACKENDS = frozenset({"mps", "metal"})


# ---------------------------------------------------------------------------
# Backend query
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the current JAX default backend name (lowercase).

    Common values: ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"mps"`` (Apple GPU
    via jax-mps / MLX).
    """
    import jax
    return jax.default_backend().lower()


def supports_float64(backend: str | None = None) -> bool:
    """Return whether *backend* supports float64/complex128 natively."""
    if backend is None:
        backend = get_backend()
    return backend.lower() not in _NO_F64_BACKENDS


# ---------------------------------------------------------------------------
# X64 policy
# ---------------------------------------------------------------------------

def enable_x64(*, quiet: bool = False) -> None:
    """Enable JAX float64 support (idempotent).

    Must be called **before** any JAX computation for the flag to
    take effect globally.
    """
    import jax
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        if not quiet:
            logger.debug("Enabled JAX x64 mode")


def is_x64_enabled() -> bool:
    """Return ``True`` if JAX x64 mode is active."""
    import jax
    return bool(jax.config.jax_enable_x64)


def require_x64(component: str) -> None:
    """Raise ``RuntimeError`` if JAX x64 is not enabled.

    Parameters
    ----------
    component : str
        Human-readable name of the component that needs x64
        (used in the error message).
    """
    if not is_x64_enabled():
        raise RuntimeError(
            f"{component} requires float64 precision, but JAX is running "
            f"in 32-bit mode (jax_enable_x64 is not set).\n\n"
            f"Enable 64-bit mode before constructing the mesh:\n"
            f"  - environment variable: JAX_ENABLE_X64=1\n"
            f"  - Python:  jax.config.update('jax_enable_x64', True)\n"
            f"  - legoESM: from legoesm.runtime import enable_x64; enable_x64()"
        )


# ---------------------------------------------------------------------------
# XLA flag helpers (from parallel.device_config)
# ---------------------------------------------------------------------------

TPU_XLA_FLAGS = {
    "xla_tpu_enable_async_collective_fusion": "true",
    "xla_tpu_enable_data_parallel_all_reduce_opt": "true",
    "xla_tpu_enable_latency_hiding_scheduler": "LHS_DEFAULT",
}

NVIDIA_GPU_XLA_FLAGS = {
    "xla_gpu_cudnn_gemm_fusion_level": "3",
    # Overlap compute with collective communication (halo exchange, allreduce).
    # NOTE: the per-collective async toggles
    # ``xla_gpu_enable_async_all_reduce`` and
    # ``xla_gpu_enable_async_collectives`` were REMOVED in newer XLA
    # (JAX ≥ ~0.5 / current 0.9–0.10).  XLA aborts with an uncatchable
    # ``LOG(FATAL) Unknown flags in XLA_FLAGS`` the instant *any* GPU
    # backend initialises if they are present — which silently broke
    # every production GPU run and the GPU scaling harness on this host.
    # Async overlap for all collectives (all-reduce, ppermute-based halo
    # exchange, all-gather) is now driven by the latency-hiding scheduler
    # below, so dropping the two obsolete flags loses no functionality on
    # current XLA.  Do NOT re-add them without gating on an XLA version
    # that still accepts them.
    "xla_gpu_enable_latency_hiding_scheduler": "true",
    "xla_gpu_enable_highest_priority_async_stream": "true",
    # Multi-shard autotune cache-sync crashes intermittently at >=64 ranks
    # ("No configuration found in cache after synchronizing results across all
    # shards"); disabling it is a no-op at <=1 shard.  A launcher that sets
    # --xla_gpu_shard_autotuning explicitly still wins (set_xla_flags dedups).
    "xla_gpu_shard_autotuning": "false",
    # CUDA Graphs / command buffers — XLA can capture sequences of
    # kernel launches and replay them as a single command buffer, which
    # eliminates the ~5-10μs per-launch overhead that dominates
    # small-grain step kernels.  Requires CUDA ≥ 12.3 (XLA falls back
    # silently on older runtimes, so always-on is safe).
    #
    # COLLECTIVES is deliberately NOT in the list.  Capturing NCCL
    # collectives into command buffers freezes the thunk schedule and
    # discards the dynamic compute/comm overlap the latency-hiding
    # scheduler provides: measured on the MPAS shard_map lane
    # (Levante, 2 nodes x 2 A100, s6@4, job 26873637) the full
    # FUSION,CUSTOM_CALL,CUBLAS,CUDNN,COLLECTIVES set was +25% step
    # time (2.81 -> 3.52 ms) versus XLA defaults; the non-collective
    # tokens measured noise-neutral in the 2026-06-15 np1 audits
    # (jobs 8488104/8490224), attributing the loss to COLLECTIVES.
    # The old FUSION,CUSTOM_CALL,COLLECTIVES default was never A/B'd
    # at multi-rank (np1 has no collectives).  Re-enable per-run for allreduce-heavy
    # data-parallel workloads with no interleaved compute (the one
    # regime where capture can win) via
    # LEGOESM_XLA_CMDBUF_COLLECTIVES=1 — set BEFORE the first
    # ``legoesm.runtime`` import: the value freezes when this module
    # loads (XLA_FLAGS must be set before JAX initialises anyway, so
    # later mutation could never take effect).
    "xla_gpu_enable_command_buffer": (
        "FUSION,CUSTOM_CALL,COLLECTIVES"
        if os.environ.get("LEGOESM_XLA_CMDBUF_COLLECTIVES") == "1"
        else "FUSION,CUSTOM_CALL"
    ),
}

AMD_GPU_XLA_FLAGS: dict[str, str] = {
    # ROCm does not use cuDNN; no vendor-specific flags needed yet.
}


def set_xla_flags(flags: dict[str, str]) -> None:
    """Append XLA flags to ``XLA_FLAGS``, without duplicating existing keys."""
    existing = os.environ.get("XLA_FLAGS", "")
    new_parts = []
    for key, value in flags.items():
        flag = f"--{key}={value}"
        if key not in existing:
            new_parts.append(flag)
    if new_parts:
        combined = (existing + " " + " ".join(new_parts)).strip()
        os.environ["XLA_FLAGS"] = combined


def detect_gpu_vendor_pre_init() -> str | None:
    """Detect GPU vendor *without* triggering ``jax.devices()``.

    Once ``jax.devices()`` runs, the XLA client is initialised and any
    further mutations of ``os.environ['XLA_FLAGS']`` no longer take
    effect.  The latency-hiding scheduler flags applied below
    (``NVIDIA_GPU_XLA_FLAGS``, ``AMD_GPU_XLA_FLAGS``) are critical for
    multi-GPU scaling, so we must set them before JAX is initialised.

    Returns ``"nvidia"`` / ``"amd"`` if a vendor is unambiguously
    identifiable from environment hints, otherwise ``None`` and the
    caller should fall back to the post-init JAX-based detection (which
    will at least pick the right matmul precision even if ``XLA_FLAGS``
    is too late to alter).
    """
    # SLURM and Nvidia commonly set ``CUDA_VISIBLE_DEVICES=-1`` or
    # ``CUDA_VISIBLE_DEVICES=NoDevFiles`` to mask GPUs on CPU-only
    # allocations.  Those values are non-empty strings, so a naive
    # truthiness check would mis-detect NVIDIA and inject GPU XLA flags
    # into the CPU backend.  Reject them explicitly.
    _CUDA_DISABLED = {"-1", "NoDevFiles", ""}

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    hip = os.environ.get("HIP_VISIBLE_DEVICES", "")
    rocr = os.environ.get("ROCR_VISIBLE_DEVICES", "")

    has_cuda = cuda not in _CUDA_DISABLED and not cuda.startswith("-")
    has_hip = (hip not in _CUDA_DISABLED and not hip.startswith("-")) or (
        rocr not in _CUDA_DISABLED and not rocr.startswith("-")
    )

    if has_cuda and not has_hip:
        return "nvidia"
    if has_hip:
        return "amd"
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms.startswith("cuda"):
        return "nvidia"
    if platforms.startswith("rocm"):
        return "amd"
    return None


def _is_nvidia_gpu() -> bool:
    """Return ``True`` if the default GPU device is an NVIDIA (CUDA) GPU."""
    import jax
    devices = jax.devices()
    if not devices:
        return False
    kind = str(getattr(devices[0], "device_kind", "")).lower()
    return "nvidia" in kind


def _is_amd_gpu() -> bool:
    """Return ``True`` if the default GPU device is an AMD (ROCm) GPU."""
    import jax
    devices = jax.devices()
    if not devices:
        return False
    kind = str(getattr(devices[0], "device_kind", "")).lower()
    return "amd" in kind or "instinct" in kind


def _cc_major(cc) -> int | None:
    """Parse the compute-capability MAJOR version from a jaxlib value.

    ``device.compute_capability`` is a CUDA-only attribute whose string
    format varies across jaxlib builds: dotted ``"8.0"`` / ``"8.6"`` /
    ``"10.0"``, packed ``"80"`` / ``"86"`` / ``"100"``, or ``"sm_80"``.
    Returns the major version, or ``None`` for anything unrecognised so the
    caller can fail CLOSED.  STRICTLY validates that the whole token is
    numeric — ``"8x"`` / ``"sm_8x"`` must NOT slip through as major 8 (the
    naive ``s[:-1]`` form did; codex 2026-06-13 MEDIUM).
    """
    if cc is None:
        return None
    s = str(cc).lower().strip()
    if s.startswith("sm_"):
        s = s[3:]
    if "." in s:
        parts = s.split(".")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None
        return int(parts[0])
    # Packed form (major+minor, no separator) needs >= 2 digits: "80"->8,
    # "100"->10.  A lone "8" is ambiguous/unreal for CUDA CC -> reject.
    if not s.isdigit() or len(s) < 2:
        return None
    return int(s[:-1])


def _nvidia_tf32_supported() -> bool:
    """Return ``True`` only if EVERY CUDA device has TF32 tensor cores.

    TF32 is an Ampere (sm_80) and later feature; Turing (RTX 8000, sm_75)
    and earlier have NO TF32 hardware, so requesting
    ``jax_default_matmul_precision="tensorfloat32"`` there is a silent
    no-op at best.  Because that config is PROCESS-GLOBAL, a heterogeneous
    fleet (e.g. ``[sm_80, sm_75]``) must NOT enable it — it would degrade
    the Turing card's matmuls incorrectly.  So gate on the MINIMUM compute
    capability across all CUDA devices being >= 8 (codex 2026-06-13 MAJOR:
    the previous ``devices()[0]``-only check missed the mixed fleet).

    Fails CLOSED (no TF32) when there are no devices, or ANY device's
    compute capability is missing or unparseable.  The call site already
    guards ``vendor == "nvidia"``, so every device here is expected to be a
    CUDA device with a parseable CC; a ``None`` (missing OR unparseable) is
    an anomaly (an unknown CUDA device alongside Ampere) that must NOT
    silently enable the process-global TF32 setting (codex 2026-06-13:
    skipping ``None`` majors was a partial-unknown fail-OPEN).
    """
    import jax
    devices = jax.devices()
    if not devices:
        return False
    majors = []
    for d in devices:
        m = _cc_major(getattr(d, "compute_capability", None))
        if m is None:
            return False  # missing/unparseable CC on an NVIDIA device
        majors.append(m)
    return min(majors) >= 8


def gpu_vendor() -> str:
    """Return the GPU vendor: ``"nvidia"``, ``"amd"``, or ``"unknown"``.

    Only meaningful when ``get_backend() == "gpu"``.
    """
    if _is_nvidia_gpu():
        return "nvidia"
    if _is_amd_gpu():
        return "amd"
    return "unknown"


def _detect_backend_pre_init() -> str | None:
    """Detect the backend without calling ``jax.default_backend()``.

    Mirrors :func:`detect_gpu_vendor_pre_init`: we cannot afford to
    initialise the PJRT client (via ``jax.default_backend()`` or
    ``jax.devices()``) before the GPU-specific XLA scheduler flags are
    set, otherwise ``XLA_FLAGS`` mutations no-op.  Returns one of
    ``"tpu"``, ``"gpu"``, ``"mps"``, ``"cpu"``, or ``None`` when no
    hint is available (caller must fall back to ``get_backend()``).
    """
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    if platforms:
        if platforms.startswith("tpu"):
            return "tpu"
        if platforms.startswith(("cuda", "rocm", "gpu")):
            return "gpu"
        if platforms.startswith("mps"):
            return "mps"
        if platforms.startswith("cpu"):
            return "cpu"
    if detect_gpu_vendor_pre_init() is not None:
        return "gpu"
    if os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
        return "tpu"
    return None


def configure_backend(backend: str | None = None) -> str:
    """Apply backend-specific XLA flags and JAX options.

    This should be called **once at startup**, before any JAX computation.
    If *backend* is ``None`` the current default backend is detected
    from environment hints (``JAX_PLATFORMS``, ``CUDA_VISIBLE_DEVICES``,
    ``TPU_NAME``).  We deliberately avoid ``jax.default_backend()`` /
    ``jax.devices()`` until **after** the GPU XLA scheduler flags are
    set — otherwise PJRT initialises with the wrong flags.

    Returns the resolved backend name (lowercase).
    """
    import jax

    if backend is None:
        backend = _detect_backend_pre_init()
    if backend is None:
        # No hint at all — fall through to the JAX default.  We still
        # try to apply NVIDIA flags pre-init in case the heuristic
        # missed something (env var weirdness).  This branch is the
        # last resort and will warn from the GPU path below if the
        # client is already up.
        backend = get_backend()
    backend = backend.lower()

    if backend == "tpu":
        set_xla_flags(TPU_XLA_FLAGS)
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        # NOTE: the legacy `jax_spmd_mode='allow_all'` toggle was removed
        # in modern JAX (0.9+) — `jax.config.update("jax_spmd_mode", ...)`
        # raises `AttributeError: Unrecognized config option: jax_spmd_mode`,
        # which would crash the canonical TPU bootstrap on multi-host pods.
        # Under the unified sharding model SPMD partitioning is automatic
        # for sharded arrays, so no explicit toggle is required.

    elif backend == "gpu":
        # Detect vendor BEFORE ``jax.devices()`` so the XLA scheduler
        # flags below land in the env var that the XLA client will
        # consume on first init.  Once ``jax.devices()`` runs, mutating
        # ``os.environ['XLA_FLAGS']`` is silently ineffective on most
        # JAX versions — the latency-hiding flags are critical for
        # multi-GPU scaling, so we cannot afford that race.
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
        # On multi-process MPI runs (the standard case for legoESM at
        # scale), several ranks share the same physical GPU under
        # ``MPS`` or co-located workers.  XLA's default
        # ``XLA_PYTHON_CLIENT_PREALLOCATE=true`` then OOMs because
        # each process tries to grab 90 % of HBM.  Disable
        # preallocation when MPI is detected; keep it on for
        # single-process runs (preallocation reduces fragmentation
        # over a long simulation).  User overrides win.
        if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
            _multi_proc = (
                int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1
                or int(os.environ.get("PMI_SIZE", "1")) > 1
                or int(os.environ.get("SLURM_NTASKS_PER_NODE", "1")) > 1
            )
            if _multi_proc:
                os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        pre_init_vendor = detect_gpu_vendor_pre_init()
        if pre_init_vendor == "nvidia":
            set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
        elif pre_init_vendor == "amd":
            set_xla_flags(AMD_GPU_XLA_FLAGS)

        devices = jax.devices()
        vendor = gpu_vendor()
        if pre_init_vendor is None:
            # Vendor was only known after JAX init; XLA_FLAGS already
            # locked in.  Still apply best-effort matmul precision
            # below, and warn so the user can pre-set ``CUDA_VISIBLE_DEVICES``
            # / ``HIP_VISIBLE_DEVICES`` for the next run.
            if vendor == "nvidia":
                set_xla_flags(NVIDIA_GPU_XLA_FLAGS)
                logger.warning(
                    "GPU vendor detected post-init; XLA scheduler flags "
                    "may not take effect this run.  Set "
                    "CUDA_VISIBLE_DEVICES or JAX_PLATFORMS=cuda before "
                    "import to enable latency-hiding flags.",
                )
            elif vendor == "amd":
                set_xla_flags(AMD_GPU_XLA_FLAGS)
                logger.warning(
                    "GPU vendor detected post-init; XLA scheduler flags "
                    "may not take effect this run.  Set "
                    "HIP_VISIBLE_DEVICES before import to enable "
                    "latency-hiding flags.",
                )

        # TensorFloat32 is an NVIDIA Ampere+ (sm_80) feature (19-bit
        # mantissa).  AMD GPUs and pre-Ampere NVIDIA (Turing sm_75, e.g.
        # RTX 8000) have NO TF32 hardware, so only enable it where the
        # compute capability actually supports it — otherwise it is a
        # silent no-op (Turing) or a mixed-fleet correctness foot-gun.
        if vendor == "nvidia" and _nvidia_tf32_supported():
            jax.config.update("jax_default_matmul_precision", "tensorfloat32")
            logger.info("TF32 matmul precision enabled (compute capability "
                        ">= 8.0)")
        logger.info("GPU vendor: %s (%d device(s))", vendor, len(devices))

    elif backend == "mps":
        # MLX manages its own device; no XLA flags or persistent JIT cache.
        _apply_mps_rev_workaround()

    else:  # cpu
        if "XLA_FLAGS" not in os.environ:
            try:
                # NOTE: the legacy `intra_op_parallelism_threads` flag is
                # NOT recognized by the current XLA parse_flags_from_env
                # and crashes at first JAX use with:
                #   F parse_flags_from_env.cc:234]
                #     Unknown flag in XLA_FLAGS: --intra_op_parallelism_threads=N
                # Drop it; rely on the default XLA thread-pool autoscaling
                # driven by `os.cpu_count()`.  `xla_cpu_multi_thread_eigen`
                # remains valid and meaningful for multi-core CPUs.
                set_xla_flags({
                    "xla_cpu_multi_thread_eigen": "true",
                })
            except Exception:
                pass
            # XLA no longer accepts ``--intra_op_parallelism_threads`` since
            # jaxlib 0.10. Eigen pulls its worker count from ``OMP_NUM_THREADS``
            # (falling back to the hardware concurrency), so set that instead.
            if "OMP_NUM_THREADS" not in os.environ:
                try:
                    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
                except Exception:
                    pass

    # JAX persistent JIT-compile cache.  Off by default in upstream
    # JAX; legoESM's compiled segment (~2600 s cold-compile, ~600 s
    # warm) is the dominant per-job overhead for AMIP / OMIP runs.
    # Re-using the XLA cache across runs cuts subsequent jobs to a
    # few seconds of cache-lookup.  Turn it on whenever the backend
    # is GPU/CPU; MPS/TPU paths often have their own compile
    # caches and we leave them alone.
    if backend in ("gpu", "cpu"):
        _configure_persistent_jit_cache()

    logger.info("Configured XLA for %s backend", backend)
    return backend


def _mps_rev_is_broken() -> bool:
    """Probe the jax-mps negative-stride-reverse mis-fusion (2026-07-21).

    On jax-mps <= 0.10.7 a 1-D ``lax.rev`` fused into an elementwise or
    scatter consumer is silently DROPPED (``u[::-1] * 2`` returns ``u * 2``;
    ``x.at[i].set(u[::-1])`` writes the un-reversed strip).  Cube-sphere
    grid construction builds its corner-metric matrices through exactly
    that pattern, so every cubed-sphere run produces garbage momentum
    tendencies.  Returns True when the platform exhibits the bug.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    u = jnp.arange(8.0, dtype=jnp.float32)
    got = np.asarray(jax.jit(lambda t: t[::-1] * 2.0)(u))
    want = np.arange(8.0, dtype=np.float32)[::-1] * 2.0
    return bool(np.abs(got - want).max() > 0)


def _apply_mps_rev_workaround() -> None:
    """Re-route ``lax.rev`` through gather on the mps platform when broken.

    Centralized fix for the fusion bug probed by :func:`_mps_rev_is_broken`:
    register a platform-specific lowering for ``rev_p`` that emits
    iota+gather (``jnp.take`` with a descending index vector) instead of the
    native reverse.  Gather-based reversal is unaffected by the fusion bug,
    so every ``[::-1]`` / ``jnp.flip`` in the codebase becomes safe without
    touching call sites.  No-op when the plugin lowers ``rev`` correctly
    (future jax-mps releases), so upgrading removes the cost automatically.
    """
    # Probe OUTSIDE the registration try (codex r2): a probe/import/JAX-init
    # failure must NOT masquerade as "rev is broken" and fail the run closed.
    # If the probe itself cannot run, we cannot claim the platform is broken —
    # let it through (any real breakage still surfaces downstream), and only
    # fail closed when the probe POSITIVELY reports broken rev but the fix
    # cannot be installed.
    try:
        broken = _mps_rev_is_broken()
    except Exception as exc:  # pragma: no cover - probe is a 2-op jit
        logger.warning(
            "jax-mps: could not probe the lax.rev fusion bug (%r); "
            "assuming the platform is healthy. If cubed-sphere runs blow up "
            "on mps, set JAX_PLATFORMS=cpu.", exc,
        )
        return
    if not broken:
        return

    try:
        from jax._src import dispatch as _dispatch  # noqa: F401 (force init)
        from jax._src.interpreters import mlir as _mlir
        from jax import lax as _lax
        import jax.numpy as jnp

        def _rev_via_gather(operand, *, dimensions):
            out = operand
            for d in dimensions:
                n = operand.shape[d]
                # descending iota, NOT [::-1] — that would re-enter rev_p
                idx = n - 1 - _lax.iota(jnp.int32, n)
                out = jnp.take(out, idx, axis=d)
            return out

        _mlir.register_lowering(
            _lax.rev_p,
            _mlir.lower_fun(_rev_via_gather, multiple_results=False),
            platform="mps",
        )
        logger.warning(
            "jax-mps: native lax.rev mis-fuses with its consumer "
            "(reversed operand silently un-reversed); re-routed rev "
            "through gather on the mps platform. Upgrade jax-mps and "
            "this workaround deactivates itself."
        )
    except Exception as exc:
        # Fail CLOSED: the probe POSITIVELY proved the platform mis-fuses rev,
        # and the fix could not be installed — continuing would produce the
        # exact garbage-momentum blowup the workaround exists to prevent.
        raise RuntimeError(
            "jax-mps native lax.rev is broken on this platform (reversed "
            "operands are silently un-reversed when fused) and the gather-"
            "lowering workaround could not be registered "
            f"({exc!r}; likely a jax private-API change). Refusing to run "
            "with known-corrupt cubed-sphere metrics — set JAX_PLATFORMS=cpu "
            "or upgrade jax-mps."
        ) from exc


def _configure_persistent_jit_cache() -> None:
    """Activate JAX's persistent JIT-compile cache.

    Respects two environment variables:

    * ``LEGOESM_JIT_CACHE_DIR`` — directory to store cached compiles.
      Defaults to ``${XDG_CACHE_HOME:-~/.cache}/legoesm/jit_cache``.
      Set to ``""`` (empty) to disable caching for this process
      without code changes (handy for clean-room benchmark runs).
    * ``LEGOESM_JIT_CACHE_MIN_SECS`` — only cache compiles slower
      than this many seconds.  Default ``1.0`` matches the JAX
      upstream convention and keeps short compiles out of the cache.

    Idempotent: a second call in the same process is a no-op.
    Safe to call before or after ``jax.devices()``.
    """
    import jax

    if getattr(_configure_persistent_jit_cache, "_done", False):
        return

    env_dir = os.environ.get("LEGOESM_JIT_CACHE_DIR", None)
    if env_dir == "":
        # Explicit opt-out.
        _configure_persistent_jit_cache._done = True  # type: ignore[attr-defined]
        return
    if env_dir is None:
        # The XDG base-directory contract requires an absolute path;
        # treat empty / relative ``XDG_CACHE_HOME`` as unset so we do
        # not silently scatter caches into the job working directory
        # on Slurm / container setups that leave the variable set to
        # a value like "".
        xdg = os.environ.get("XDG_CACHE_HOME") or ""
        if not xdg or not os.path.isabs(xdg):
            xdg = os.path.join(os.path.expanduser("~"), ".cache")
        cache_dir = os.path.join(xdg, "legoesm", "jit_cache")
    else:
        cache_dir = os.path.expanduser(env_dir)

    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not create JIT cache dir %s (%s); persistent cache "
            "disabled this run.", cache_dir, exc,
        )
        _configure_persistent_jit_cache._done = True  # type: ignore[attr-defined]
        return

    try:
        min_secs = float(os.environ.get("LEGOESM_JIT_CACHE_MIN_SECS", "1.0"))
    except ValueError:
        min_secs = 1.0

    # Catch only the narrow class of "this JAX build does not know
    # about the persistent-cache options" failures.  Anything else
    # (e.g. a typed-argument programming error introduced by a future
    # refactor) must propagate so the regression is visible — this
    # whole helper exists *because* the cache being silently off is
    # the regression we are trying to fix.  ``LEGOESM_JIT_CACHE_DIR=""``
    # is the explicit escape hatch for users on older JAX.
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # ``jax_persistent_cache_min_entry_size_bytes`` defaults to 0 in
        # current JAX (cache everything that meets the time threshold).
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", min_secs,
        )
    except AttributeError as exc:
        # JAX < 0.4.18 (or a vendored fork) is missing these options
        # entirely.  Skip with a single warning so legoESM still runs;
        # the user can drop the persistent-cache flags by setting
        # LEGOESM_JIT_CACHE_DIR="".
        logger.warning(
            "JAX persistent cache options not available in this JAX "
            "build (%s); legoESM will run without compile caching.  "
            "Upgrade JAX to 0.4.18+ or set LEGOESM_JIT_CACHE_DIR=\"\" "
            "to silence this warning.", exc,
        )
        _configure_persistent_jit_cache._done = True  # type: ignore[attr-defined]
        return

    logger.info(
        "JAX persistent JIT cache: dir=%s, min_compile_secs=%.1f",
        cache_dir, min_secs,
    )
    _configure_persistent_jit_cache._done = True  # type: ignore[attr-defined]


def enable_persistent_compile_cache() -> None:
    """Public, side-effect-only entrypoint for the persistent XLA compile cache.

    Drivers that do NOT go through ``configure_backend`` / ``bootstrap`` —
    ``scripts/run/run_amip.py`` and the ``training`` package — still want the
    cross-process JIT cache.  The RRTMGP radiation graph cold-compiles in
    ~2600 s; without the cache that cost is re-paid on *every* process launch
    (each iterate/debug run, each restart, each sweep point).  With the cache
    on, the second and later launches are a few-second cache lookup.

    Thin wrapper over :func:`_configure_persistent_jit_cache` so the private
    symbol is never imported across packages (CI ratchet
    ``test_no_private_cross_imports``).  Idempotent; call before the first
    ``jax.jit`` compile in the process.  Honors ``LEGOESM_JIT_CACHE_DIR`` /
    ``LEGOESM_JIT_CACHE_MIN_SECS`` (set the dir to ``""`` to opt out).
    """
    _configure_persistent_jit_cache()


# ---------------------------------------------------------------------------
# Spectral-backend guard (from core.hardware)
# ---------------------------------------------------------------------------

def check_spectral_backend(*, allow_unsupported: bool = False) -> None:
    """Verify the current backend supports float64/complex128.

    The spectral solver requires these types.  Backends like Apple MPS
    (MLX) do not provide them and will produce incorrect results.

    Raises ``ValueError`` unless *allow_unsupported* is ``True``
    (in which case a warning is emitted instead).
    """
    import jax

    if not jax.config.jax_enable_x64:
        msg = (
            "The spectral solver requires float64 and complex128 arithmetic, "
            "but JAX is running in 32-bit mode (jax_enable_x64 is not set).\n\n"
            "Remediation: set the environment variable JAX_ENABLE_X64=True "
            "or call jax.config.update('jax_enable_x64', True) before "
            "importing any spectral modules."
        )
        if allow_unsupported:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        else:
            raise ValueError(msg)

    backend = get_backend()
    if backend.lower() not in _NO_F64_BACKENDS:
        return

    msg = (
        f"The spectral solver requires float64 and complex128, "
        f"but the current backend is '{backend}', which does not "
        f"support these types.\n\n"
        f"Remediation options:\n"
        f"  1. Force CPU: JAX_PLATFORMS=cpu python your_script.py\n"
        f"  2. Use the finite-volume solver (cubed-sphere), which "
        f"works in float32 on all backends.\n"
        f"  3. Set allow_unsupported=true to bypass (expert only)."
    )
    if allow_unsupported:
        warnings.warn(
            f"Spectral solver on unsupported backend '{backend}'. "
            f"Results may be incorrect.\n\n" + msg,
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise ValueError(msg)
